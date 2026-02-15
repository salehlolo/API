#!/usr/bin/env python3
"""
README (Quick Start)
====================
This script fetches BTC/USDT spot price from Binance and supports 3 modes via env var MODE:
- PRICE_ONLY: fetch validated price and send it to Telegram.
- PAPER (default): generate MA crossover signals + simulate trades + track PnL.
- LIVE: safety mode; real Binance order execution is intentionally disabled.

Requirements
------------
- Python 3.11+
- requests
- Optional: python-dotenv (for local .env loading)

Install
-------
1) Create and activate a virtual environment (recommended):
   python3.11 -m venv .venv
   source .venv/bin/activate

2) Install dependencies:
   pip install requests python-dotenv

Environment Variables
---------------------
Required (for Telegram send):
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID

Trading config (optional):
- MODE=PRICE_ONLY | PAPER | LIVE  (default PAPER)
- INITIAL_USDT_BALANCE=1000
- TRADE_SIZE_PCT=0.2              (20% of available USDT on BUY)
- FEE_RATE=0.001                  (0.1% fee per trade)
- FAST_MA=10
- SLOW_MA=30
- PRICE_WINDOW_SIZE=200
- STATUS_INTERVAL_SECONDS=600     (10 minutes)

You can use .env file (if python-dotenv is installed):
   MODE=PAPER
   TELEGRAM_BOT_TOKEN=123456:ABCDEF...
   TELEGRAM_CHAT_ID=123456789

How to get TELEGRAM_CHAT_ID
---------------------------
1) Send any message to your bot.
2) Call:
   https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/getUpdates
3) Extract from latest update:
   message.chat.id

Run Examples
------------
- Run once:
  python btc_price_to_telegram.py

- Run every 60 seconds:
  python btc_price_to_telegram.py --interval 60

- Test Binance fetch + validation only (no Telegram send):
  python btc_price_to_telegram.py --test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import requests
from requests import Response
from requests.exceptions import RequestException, Timeout
from zoneinfo import ZoneInfo

# Optional .env support.
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"
BINANCE_TIME_URL = "https://api.binance.com/api/v3/time"
BINANCE_SYMBOL = "BTCUSDT"
CAIRO_TZ = ZoneInfo("Africa/Cairo")

CONNECT_TIMEOUT_SECONDS = 5
READ_TIMEOUT_SECONDS = 10
REQUEST_TIMEOUT = (CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 20.0
JITTER_SECONDS = 0.5
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}

MODE_VALUES = {"PRICE_ONLY", "PAPER", "LIVE"}
PRICES_LOG_PATH = Path("prices_log.csv")
TRADES_LOG_PATH = Path("trades.csv")


@dataclass(frozen=True)
class PriceData:
    symbol: str
    price: Decimal


@dataclass
class AppConfig:
    mode: str
    initial_usdt_balance: Decimal
    trade_size_pct: Decimal
    fee_rate: Decimal
    fast_ma: int
    slow_ma: int
    price_window_size: int
    status_interval_seconds: int


@dataclass
class PaperState:
    balance_usdt: Decimal
    balance_btc: Decimal
    entry_price: Decimal | None
    realized_pnl: Decimal
    last_signal: str
    last_summary_ts: float


class PaperExecutor:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.prices: deque[Decimal] = deque(maxlen=max(cfg.price_window_size, cfg.slow_ma + 5))
        self.state = PaperState(
            balance_usdt=cfg.initial_usdt_balance,
            balance_btc=Decimal("0"),
            entry_price=None,
            realized_pnl=Decimal("0"),
            last_signal="NONE",
            last_summary_ts=time.time(),
        )
        self._ensure_csv_headers()

    def _ensure_csv_headers(self) -> None:
        if not PRICES_LOG_PATH.exists():
            with PRICES_LOG_PATH.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "price"])

        if not TRADES_LOG_PATH.exists():
            with TRADES_LOG_PATH.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "side",
                    "price",
                    "qty",
                    "fee",
                    "balance_usdt",
                    "balance_btc",
                    "realized_pnl",
                ])

    def log_price(self, ts: str, price: Decimal) -> None:
        with PRICES_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([ts, f"{price}"])

    def ma(self, period: int, values: list[Decimal]) -> Decimal:
        if len(values) < period:
            raise ValueError("Not enough values for moving average")
        chunk = values[-period:]
        return sum(chunk) / Decimal(period)

    def evaluate_signal(self) -> str:
        values = list(self.prices)
        if len(values) < self.cfg.slow_ma + 1:
            return "HOLD"

        prev_values = values[:-1]
        fast_prev = self.ma(self.cfg.fast_ma, prev_values)
        slow_prev = self.ma(self.cfg.slow_ma, prev_values)
        fast_now = self.ma(self.cfg.fast_ma, values)
        slow_now = self.ma(self.cfg.slow_ma, values)

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now

        if crossed_up:
            return "BUY"
        if crossed_down:
            return "SELL"
        return "HOLD"

    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        if self.state.balance_btc <= 0 or self.state.entry_price is None:
            return Decimal("0")
        return (current_price - self.state.entry_price) * self.state.balance_btc

    def execute(self, signal: str, current_price: Decimal, ts: str) -> dict[str, Any] | None:
        if signal == "BUY":
            if self.state.balance_btc > 0:
                return None

            usdt_to_use = self.state.balance_usdt * self.cfg.trade_size_pct
            if usdt_to_use <= 0:
                return None

            fee_usdt = usdt_to_use * self.cfg.fee_rate
            net_usdt = usdt_to_use - fee_usdt
            qty_btc = net_usdt / current_price

            self.state.balance_usdt -= usdt_to_use
            self.state.balance_btc += qty_btc
            self.state.entry_price = current_price
            self.state.last_signal = "BUY"

            self._log_trade(ts, "BUY", current_price, qty_btc, fee_usdt)
            return {
                "side": "BUY",
                "price": current_price,
                "qty": qty_btc,
                "fee": fee_usdt,
                "realized_pnl": self.state.realized_pnl,
            }

        if signal == "SELL":
            qty_btc = self.state.balance_btc
            if qty_btc <= 0:
                return None

            gross_usdt = qty_btc * current_price
            fee_usdt = gross_usdt * self.cfg.fee_rate
            net_usdt = gross_usdt - fee_usdt

            entry = self.state.entry_price if self.state.entry_price is not None else current_price
            cost_basis = qty_btc * entry
            trade_pnl = net_usdt - cost_basis
            self.state.realized_pnl += trade_pnl

            self.state.balance_usdt += net_usdt
            self.state.balance_btc = Decimal("0")
            self.state.entry_price = None
            self.state.last_signal = "SELL"

            self._log_trade(ts, "SELL", current_price, qty_btc, fee_usdt)
            return {
                "side": "SELL",
                "price": current_price,
                "qty": qty_btc,
                "fee": fee_usdt,
                "realized_pnl": self.state.realized_pnl,
                "trade_pnl": trade_pnl,
            }

        return None

    def _log_trade(self, ts: str, side: str, price: Decimal, qty: Decimal, fee: Decimal) -> None:
        with TRADES_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ts,
                side,
                f"{price}",
                f"{qty}",
                f"{fee}",
                f"{self.state.balance_usdt}",
                f"{self.state.balance_btc}",
                f"{self.state.realized_pnl}",
            ])


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def maybe_load_dotenv() -> None:
    if load_dotenv is None:
        logging.info("python-dotenv not installed; skipping .env loading.")
        return

    loaded = load_dotenv()
    if loaded:
        logging.info("Loaded environment variables from .env")
    else:
        logging.info("No .env file found (or file had no variables).")


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: tuple[float, float],
    json_payload: dict[str, Any] | None = None,
) -> Response:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.request(method=method, url=url, timeout=timeout, json=json_payload)

            if response.status_code in TRANSIENT_STATUSES:
                logging.warning(
                    "Transient HTTP %s from %s on attempt %d/%d",
                    response.status_code,
                    url,
                    attempt,
                    MAX_RETRIES,
                )
                if attempt == MAX_RETRIES:
                    return response
                sleep_with_backoff(attempt)
                continue

            return response

        except (Timeout, RequestException) as exc:
            last_error = exc
            logging.warning(
                "Request error on attempt %d/%d for %s: %s",
                attempt,
                MAX_RETRIES,
                url,
                exc,
            )
            if attempt == MAX_RETRIES:
                break
            sleep_with_backoff(attempt)

    raise RuntimeError(f"HTTP request failed after {MAX_RETRIES} attempts: {url}") from last_error


def sleep_with_backoff(attempt: int) -> None:
    backoff = min(INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
    jitter = random.uniform(0, JITTER_SECONDS)
    delay = backoff + jitter
    logging.info("Retrying in %.2f seconds...", delay)
    time.sleep(delay)


def parse_json_response(response: Response, endpoint_name: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise ValueError(f"{endpoint_name} returned non-JSON response") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{endpoint_name} JSON payload is not an object")

    return payload


def validate_binance_connectivity(session: requests.Session) -> int:
    response = request_with_retry(session, "GET", BINANCE_TIME_URL, timeout=REQUEST_TIMEOUT)

    if response.status_code != 200:
        raise ValueError(f"Connectivity check failed: HTTP {response.status_code}")

    payload = parse_json_response(response, "Binance time endpoint")
    if "serverTime" not in payload:
        raise ValueError("Connectivity check failed: missing 'serverTime' key")

    server_time = payload["serverTime"]
    if not isinstance(server_time, int):
        raise ValueError("Connectivity check failed: 'serverTime' is not an integer")

    return server_time


def fetch_and_validate_price(session: requests.Session) -> PriceData:
    response = request_with_retry(
        session,
        "GET",
        f"{BINANCE_PRICE_URL}?symbol={BINANCE_SYMBOL}",
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        raise ValueError(f"Price endpoint failed: HTTP {response.status_code}")

    payload = parse_json_response(response, "Binance ticker endpoint")

    if "symbol" not in payload or "price" not in payload:
        raise ValueError("Price validation failed: missing 'symbol' or 'price'")

    symbol = payload["symbol"]
    if symbol != BINANCE_SYMBOL:
        raise ValueError(f"Price validation failed: expected {BINANCE_SYMBOL}, got {symbol!r}")

    try:
        price = Decimal(str(payload["price"]))
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"Price validation failed: invalid price {payload['price']!r}") from exc

    if price <= 0:
        raise ValueError(f"Price validation failed: price must be > 0, got {price}")

    return PriceData(symbol=symbol, price=price)


def get_cairo_timestamp() -> str:
    return datetime.now(CAIRO_TZ).strftime("%Y-%m-%d %H:%M:%S")


def build_price_only_message(price_data: PriceData) -> str:
    return f"BTC/USDT: {price_data.price:.2f}\nTime (Africa/Cairo): {get_cairo_timestamp()}"


def get_telegram_config() -> tuple[str, str]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN environment variable")
    if not chat_id:
        raise ValueError("Missing TELEGRAM_CHAT_ID environment variable")
    return token, chat_id


def send_telegram_message(session: requests.Session, token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}

    response = request_with_retry(session, "POST", url, timeout=REQUEST_TIMEOUT, json_payload=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Telegram sendMessage failed: HTTP {response.status_code} | {response.text}")

    result = parse_json_response(response, "Telegram sendMessage endpoint")
    if result.get("ok") is not True:
        raise RuntimeError(f"Telegram API error payload: {result}")


def safe_send_error_to_telegram(session: requests.Session, err_msg: str) -> None:
    try:
        token, chat_id = get_telegram_config()
    except ValueError as cfg_err:
        logging.error("Cannot send error to Telegram: %s", cfg_err)
        return

    try:
        send_telegram_message(session, token, chat_id, f"[ERROR] BTC bot failed: {err_msg}")
        logging.info("Error message sent to Telegram.")
    except Exception as exc:  # noqa: BLE001
        logging.error("Failed to send error to Telegram: %s", exc)


def parse_decimal_env(name: str, default: str) -> Decimal:
    raw = os.getenv(name, default).strip()
    try:
        return Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal env value for {name}: {raw!r}") from exc


def parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer env value for {name}: {raw!r}") from exc


def load_app_config() -> AppConfig:
    mode = os.getenv("MODE", "PAPER").strip().upper() or "PAPER"
    if mode not in MODE_VALUES:
        raise ValueError(f"Invalid MODE={mode!r}. Allowed: PRICE_ONLY, PAPER, LIVE")

    cfg = AppConfig(
        mode=mode,
        initial_usdt_balance=parse_decimal_env("INITIAL_USDT_BALANCE", "1000"),
        trade_size_pct=parse_decimal_env("TRADE_SIZE_PCT", "0.2"),
        fee_rate=parse_decimal_env("FEE_RATE", "0.001"),
        fast_ma=parse_int_env("FAST_MA", 10),
        slow_ma=parse_int_env("SLOW_MA", 30),
        price_window_size=parse_int_env("PRICE_WINDOW_SIZE", 200),
        status_interval_seconds=parse_int_env("STATUS_INTERVAL_SECONDS", 600),
    )

    if cfg.fast_ma <= 0 or cfg.slow_ma <= 0:
        raise ValueError("FAST_MA and SLOW_MA must be > 0")
    if cfg.fast_ma >= cfg.slow_ma:
        raise ValueError("FAST_MA must be smaller than SLOW_MA")
    if cfg.trade_size_pct <= 0 or cfg.trade_size_pct > 1:
        raise ValueError("TRADE_SIZE_PCT must be within (0, 1]")
    if cfg.fee_rate < 0:
        raise ValueError("FEE_RATE must be >= 0")
    if cfg.price_window_size < 200:
        raise ValueError("PRICE_WINDOW_SIZE must be >= 200")
    if cfg.status_interval_seconds <= 0:
        raise ValueError("STATUS_INTERVAL_SECONDS must be > 0")

    return cfg


def process_price_only(
    session: requests.Session,
    price_data: PriceData,
    test_mode: bool,
) -> int:
    if test_mode:
        logging.info("--test mode: PRICE_ONLY message preview:\n%s", build_price_only_message(price_data))
        return 0

    token, chat_id = get_telegram_config()
    send_telegram_message(session, token, chat_id, build_price_only_message(price_data))
    logging.info("PRICE_ONLY message sent to Telegram.")
    return 0


def maybe_send_paper_status(
    session: requests.Session,
    cfg: AppConfig,
    executor: PaperExecutor,
    current_price: Decimal,
    now_ts: float,
    test_mode: bool,
) -> None:
    if now_ts - executor.state.last_summary_ts < cfg.status_interval_seconds:
        return

    unrealized = executor.unrealized_pnl(current_price)
    msg = (
        "PAPER STATUS\n"
        f"Time (Africa/Cairo): {get_cairo_timestamp()}\n"
        f"USDT: {executor.state.balance_usdt:.4f}\n"
        f"BTC: {executor.state.balance_btc:.8f}\n"
        f"Realized PnL: {executor.state.realized_pnl:.4f} USDT\n"
        f"Unrealized PnL: {unrealized:.4f} USDT"
    )

    if test_mode:
        logging.info("--test mode: status summary preview:\n%s", msg)
    else:
        token, chat_id = get_telegram_config()
        send_telegram_message(session, token, chat_id, msg)
        logging.info("PAPER status summary sent to Telegram.")

    executor.state.last_summary_ts = now_ts


def process_paper(
    session: requests.Session,
    cfg: AppConfig,
    executor: PaperExecutor,
    price_data: PriceData,
    test_mode: bool,
) -> int:
    ts = get_cairo_timestamp()
    now_ts = time.time()

    executor.prices.append(price_data.price)
    executor.log_price(ts, price_data.price)

    signal = executor.evaluate_signal()
    trade = executor.execute(signal, price_data.price, ts)

    if trade:
        unrealized = executor.unrealized_pnl(price_data.price)
        msg = (
            f"PAPER {trade['side']}\n"
            f"Price: {price_data.price:.2f}\n"
            f"Qty BTC: {trade['qty']:.8f}\n"
            f"Fee: {trade['fee']:.6f} USDT\n"
            f"USDT: {executor.state.balance_usdt:.4f}\n"
            f"BTC: {executor.state.balance_btc:.8f}\n"
            f"Realized PnL: {executor.state.realized_pnl:.4f} USDT\n"
            f"Unrealized PnL: {unrealized:.4f} USDT\n"
            f"Time (Africa/Cairo): {ts}"
        )

        if test_mode:
            logging.info("--test mode: trade notification preview:\n%s", msg)
        else:
            token, chat_id = get_telegram_config()
            send_telegram_message(session, token, chat_id, msg)
            logging.info("PAPER trade message sent to Telegram.")

    maybe_send_paper_status(session, cfg, executor, price_data.price, now_ts, test_mode)

    return 0


def run_once(
    session: requests.Session,
    cfg: AppConfig,
    test_mode: bool,
    paper_executor: PaperExecutor | None,
) -> int:
    try:
        server_time = validate_binance_connectivity(session)
        logging.info("Connectivity check passed. Binance serverTime=%s", server_time)

        price_data = fetch_and_validate_price(session)
        logging.info("Price validated: %s %s", price_data.symbol, price_data.price)

        if cfg.mode == "LIVE":
            logging.error("LIVE mode not enabled for safety. Exiting.")
            return 1

        if cfg.mode == "PRICE_ONLY":
            return process_price_only(session, price_data, test_mode)

        if paper_executor is None:
            raise RuntimeError("Paper executor is not initialized")

        return process_paper(session, cfg, paper_executor, price_data, test_mode)

    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        logging.error("Run failed: %s", err)
        if not test_mode:
            safe_send_error_to_telegram(session, err)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTC/USDT Binance fetch + Telegram + paper trading")
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Run every N seconds until interrupted (default: run once)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test Binance fetch/validation and local logic only; no Telegram sends.",
    )
    args = parser.parse_args()

    if args.interval < 0:
        parser.error("--interval must be >= 0")

    return args


def main() -> int:
    configure_logging()
    maybe_load_dotenv()

    try:
        cfg = load_app_config()
    except Exception as exc:  # noqa: BLE001
        logging.error("Config error: %s", exc)
        return 1

    args = parse_args()
    logging.info("Starting in MODE=%s", cfg.mode)

    paper_executor = PaperExecutor(cfg) if cfg.mode == "PAPER" else None

    with requests.Session() as session:
        if args.interval == 0:
            return run_once(session, cfg, args.test, paper_executor)

        logging.info("Periodic mode enabled: interval=%s sec", args.interval)
        exit_code = 0
        try:
            while True:
                exit_code = run_once(session, cfg, args.test, paper_executor)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logging.info("Interrupted by user. Exiting.")
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
