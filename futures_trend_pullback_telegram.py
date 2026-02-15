#!/usr/bin/env python3
"""
Unified Binance Futures Trend-Pullback Backtester + Telegram Notifier
=====================================================================

This is a paper-trading/backtesting script (NO live order placement).
It combines:
1) Futures data acquisition from Binance REST API
2) Trend Pullback strategy simulation
3) Fee-aware PnL/accounting + CSV trade logs
4) Telegram notifications per closed trade + final summary

Major defaults preserved from previous scripts:
- Symbol: BTCUSDT
- Interval: 1h
- EMA fast/trend: 20 / 50
- RSI: 14 (optional filter, enabled by default)
- ATR: 14
- Volume MA: 20
- ATR stop/take-profit: 1x ATR / 1.5x ATR
- Initial balance: 100 USD
- Trade allocation: 90 USD per trade
- Commission: 0.04% per side (fee_rate=0.0004)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests
from requests import Response
from requests.exceptions import RequestException, Timeout

# Optional local .env support
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

# -----------------------------
# API constants and HTTP policy
# -----------------------------
FUTURES_BASE = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"
BOOK_TICKER_ENDPOINT = "/fapi/v1/ticker/bookTicker"

TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
REQUEST_TIMEOUT = (5, 15)  # connect, read
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 20.0
JITTER_MAX = 0.5


# -----------------------------
# Data models
# -----------------------------
@dataclass
class Config:
    # Market/data
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    limit: int = 500
    start_time_ms: int | None = None
    end_time_ms: int | None = None

    # Strategy defaults (preserved)
    ema_fast: int = 20
    ema_trend: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    volume_ma_period: int = 20
    use_rsi_filter: bool = True
    rsi_threshold: float = 50.0
    max_spread: float = 2.0

    # Exits
    atr_sl_mult: float = 1.0
    rr_ratio: float = 1.5
    use_trailing_stop: bool = False
    trailing_atr_mult: float = 1.0

    # Account/risk
    initial_balance: float = 100.0
    trade_size_usd: float = 90.0
    fee_rate: float = 0.0004  # 0.04% per side
    max_concurrent_positions: int = 1

    # Output/notifications
    output_csv: str = "backtest_results.csv"
    send_telegram: bool = True


@dataclass
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


@dataclass
class Position:
    direction: str  # LONG | SHORT
    entry_time: int
    entry_price: float
    qty: float
    stop_loss: float
    take_profit: float


@dataclass
class Trade:
    timestamp: int
    entry_price: float
    exit_price: float
    position_size: float
    direction: str
    gross_pnl: float
    fees: float
    net_pnl: float
    balance: float
    exit_reason: str


# -----------------------------
# Configuration and logging
# -----------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def maybe_load_dotenv() -> None:
    if load_dotenv is None:
        logging.info("python-dotenv not installed; skipping .env loading.")
        return
    if load_dotenv():
        logging.info("Loaded environment variables from .env")


def parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw and raw.strip() else default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw and raw.strip() else default


def load_env_config() -> Config:
    return Config(
        symbol=os.getenv("SYMBOL", "BTCUSDT"),
        interval=os.getenv("INTERVAL", "1h"),
        limit=env_int("LIMIT", 500),
        start_time_ms=int(os.getenv("START_TIME_MS")) if os.getenv("START_TIME_MS") else None,
        end_time_ms=int(os.getenv("END_TIME_MS")) if os.getenv("END_TIME_MS") else None,
        ema_fast=env_int("EMA_FAST", 20),
        ema_trend=env_int("EMA_TREND", 50),
        rsi_period=env_int("RSI_PERIOD", 14),
        atr_period=env_int("ATR_PERIOD", 14),
        volume_ma_period=env_int("VOLUME_MA_PERIOD", 20),
        use_rsi_filter=parse_bool(os.getenv("USE_RSI_FILTER"), True),
        rsi_threshold=env_float("RSI_THRESHOLD", 50.0),
        max_spread=env_float("MAX_SPREAD", 2.0),
        atr_sl_mult=env_float("ATR_SL_MULT", 1.0),
        rr_ratio=env_float("RR_RATIO", 1.5),
        use_trailing_stop=parse_bool(os.getenv("USE_TRAILING_STOP"), False),
        trailing_atr_mult=env_float("TRAILING_ATR_MULT", 1.0),
        initial_balance=env_float("INITIAL_BALANCE", 100.0),
        trade_size_usd=env_float("TRADE_SIZE_USD", 90.0),
        fee_rate=env_float("FEE_RATE", 0.0004),
        max_concurrent_positions=env_int("MAX_CONCURRENT_POSITIONS", 1),
        output_csv=os.getenv("OUTPUT_CSV", "backtest_results.csv"),
        send_telegram=parse_bool(os.getenv("SEND_TELEGRAM"), True),
    )


def load_json_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("JSON config must be an object")
    return payload


def merge_config(base: Config, updates: dict[str, Any]) -> Config:
    allowed = set(Config.__dataclass_fields__.keys())
    for key in updates:
        if key not in allowed:
            raise ValueError(f"Unknown config key: {key}")
    return Config(**{**base.__dict__, **updates})


def validate_config(cfg: Config) -> None:
    if cfg.limit < 120:
        raise ValueError("limit must be >= 120")
    if cfg.initial_balance <= 0:
        raise ValueError("initial_balance must be > 0")
    if cfg.trade_size_usd <= 0:
        raise ValueError("trade_size_usd must be > 0")
    if cfg.fee_rate < 0:
        raise ValueError("fee_rate must be >= 0")
    if cfg.max_concurrent_positions <= 0:
        raise ValueError("max_concurrent_positions must be > 0")
    if cfg.ema_fast <= 1 or cfg.ema_trend <= 1 or cfg.ema_fast >= cfg.ema_trend:
        raise ValueError("EMA settings invalid")
    if cfg.rsi_period <= 1 or cfg.atr_period <= 1 or cfg.volume_ma_period <= 1:
        raise ValueError("Indicator periods must be > 1")
    if cfg.max_spread < 0:
        raise ValueError("max_spread must be >= 0")
    if cfg.atr_sl_mult <= 0 or cfg.rr_ratio <= 0:
        raise ValueError("atr_sl_mult and rr_ratio must be > 0")
    if cfg.use_trailing_stop and cfg.trailing_atr_mult <= 0:
        raise ValueError("trailing_atr_mult must be > 0 when trailing stop enabled")


# -----------------------------
# HTTP / API
# -----------------------------
def sleep_backoff(attempt: int) -> None:
    delay = min(INITIAL_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF) + random.uniform(0, JITTER_MAX)
    logging.info("Retrying in %.2f sec", delay)
    time.sleep(delay)


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_payload: dict[str, Any] | None = None,
) -> Response:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logging.info("HTTP %s %s params=%s", method, url, params)
            resp = session.request(method=method, url=url, params=params, json=json_payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code in TRANSIENT_STATUSES:
                logging.warning(
                    "Transient HTTP %s for %s (attempt %d/%d)",
                    resp.status_code,
                    url,
                    attempt,
                    MAX_RETRIES,
                )
                if attempt == MAX_RETRIES:
                    return resp
                sleep_backoff(attempt)
                continue
            return resp
        except (Timeout, RequestException) as exc:
            last_exc = exc
            logging.warning("Request error for %s (attempt %d/%d): %s", url, attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                break
            sleep_backoff(attempt)
    raise RuntimeError(f"Request failed after {MAX_RETRIES} attempts: {url}") from last_exc


def fetch_klines(session: requests.Session, cfg: Config) -> list[Candle]:
    """Fetch historical futures candlesticks from /fapi/v1/klines."""
    url = FUTURES_BASE + KLINES_ENDPOINT
    candles: list[Candle] = []
    remaining = cfg.limit
    start = cfg.start_time_ms

    while remaining > 0:
        chunk = min(remaining, 1000)
        params: dict[str, Any] = {"symbol": cfg.symbol, "interval": cfg.interval, "limit": chunk}
        if start is not None:
            params["startTime"] = start
        if cfg.end_time_ms is not None:
            params["endTime"] = cfg.end_time_ms

        resp = request_with_retry(session, "GET", url, params=params)
        if resp.status_code != 200:
            raise RuntimeError(f"Klines request failed: HTTP {resp.status_code} - {resp.text}")

        payload = resp.json()
        if not isinstance(payload, list):
            raise RuntimeError("Invalid kline payload format")
        if not payload:
            break

        parsed = [
            Candle(
                open_time=int(x[0]),
                open=float(x[1]),
                high=float(x[2]),
                low=float(x[3]),
                close=float(x[4]),
                volume=float(x[5]),
                close_time=int(x[6]),
            )
            for x in payload
        ]

        candles.extend(parsed)
        remaining = cfg.limit - len(candles)
        if len(parsed) < chunk:
            break
        start = parsed[-1].close_time + 1

    if not candles:
        raise RuntimeError("No futures candles returned")
    logging.info("Fetched %d futures candles", len(candles))
    return candles[: cfg.limit]


def fetch_spread(session: requests.Session, symbol: str) -> float:
    """Fetch current futures spread from /fapi/v1/ticker/bookTicker."""
    url = FUTURES_BASE + BOOK_TICKER_ENDPOINT
    resp = request_with_retry(session, "GET", url, params={"symbol": symbol})
    if resp.status_code != 200:
        raise RuntimeError(f"BookTicker request failed: HTTP {resp.status_code} - {resp.text}")

    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid bookTicker payload")

    bid = float(payload["bidPrice"])
    ask = float(payload["askPrice"])
    spread = ask - bid
    if spread < 0:
        raise RuntimeError(f"Invalid negative spread: {spread}")
    return spread


# -----------------------------
# Indicators
# -----------------------------
def ema(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    alpha = 2.0 / (period + 1)
    prev = sum(values[:period]) / period
    out[period - 1] = prev
    for i in range(period, len(values)):
        prev = alpha * values[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def sma(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    acc = sum(values[:period])
    out[period - 1] = acc / period
    for i in range(period, len(values)):
        acc += values[i] - values[i - period]
        out[i] = acc / period
    return out


def rsi(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return out

    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, period + 1):
        d = values[i] - values[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    def calc(g: float, l: float) -> float:
        if l == 0:
            return 100.0
        rs = g / l
        return 100.0 - (100.0 / (1 + rs))

    out[period] = calc(avg_gain, avg_loss)

    for i in range(period + 1, len(values)):
        d = values[i] - values[i - 1]
        gain = max(d, 0.0)
        loss = max(-d, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        out[i] = calc(avg_gain, avg_loss)

    return out


def atr(candles: list[Candle], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(candles)
    if len(candles) <= period:
        return out

    trs: list[float] = []
    for i, c in enumerate(candles):
        if i == 0:
            tr = c.high - c.low
        else:
            prev_close = candles[i - 1].close
            tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
        trs.append(tr)

    prev = sum(trs[1 : period + 1]) / period
    out[period] = prev
    for i in range(period + 1, len(candles)):
        prev = ((prev * (period - 1)) + trs[i]) / period
        out[i] = prev

    return out


# -----------------------------
# Telegram + utility helpers
# -----------------------------
def telegram_credentials() -> tuple[str, str]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
    return token, chat_id


def send_telegram_message(session: requests.Session, text: str) -> None:
    token, chat_id = telegram_credentials()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = request_with_retry(session, "POST", url, json_payload={"chat_id": chat_id, "text": text})
    if resp.status_code != 200:
        raise RuntimeError(f"Telegram sendMessage failed: HTTP {resp.status_code} - {resp.text}")
    payload = resp.json()
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError(f"Telegram API error payload: {payload}")


def fmt_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def max_drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def write_trades_csv(path: str, trades: list[Trade]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp",
                "entry_price",
                "exit_price",
                "position_size",
                "trade_direction",
                "gross_profit_loss",
                "fees",
                "net_profit_loss",
                "account_balance",
                "exit_reason",
            ]
        )
        for t in trades:
            w.writerow(
                [
                    fmt_ts(t.timestamp),
                    f"{t.entry_price:.8f}",
                    f"{t.exit_price:.8f}",
                    f"{t.position_size:.8f}",
                    t.direction,
                    f"{t.gross_pnl:.8f}",
                    f"{t.fees:.8f}",
                    f"{t.net_pnl:.8f}",
                    f"{t.balance:.8f}",
                    t.exit_reason,
                ]
            )


# -----------------------------
# Backtest core logic
# -----------------------------
def evaluate_exit(position: Position, candle: Candle, fee_rate: float, balance: float) -> tuple[Trade | None, float]:
    exit_price = None
    reason = ""

    if position.direction == "LONG":
        sl_hit = candle.low <= position.stop_loss
        tp_hit = candle.high >= position.take_profit
        if sl_hit and tp_hit:
            exit_price, reason = position.stop_loss, "SL_and_TP_same_candle_assume_SL"
        elif sl_hit:
            exit_price, reason = position.stop_loss, "SL"
        elif tp_hit:
            exit_price, reason = position.take_profit, "TP"
    else:
        sl_hit = candle.high >= position.stop_loss
        tp_hit = candle.low <= position.take_profit
        if sl_hit and tp_hit:
            exit_price, reason = position.stop_loss, "SL_and_TP_same_candle_assume_SL"
        elif sl_hit:
            exit_price, reason = position.stop_loss, "SL"
        elif tp_hit:
            exit_price, reason = position.take_profit, "TP"

    if exit_price is None:
        return None, balance

    entry_notional = position.entry_price * position.qty
    exit_notional = exit_price * position.qty
    fees = (entry_notional * fee_rate) + (exit_notional * fee_rate)

    gross = (
        (exit_price - position.entry_price) * position.qty
        if position.direction == "LONG"
        else (position.entry_price - exit_price) * position.qty
    )
    net = gross - fees
    balance += net

    return (
        Trade(
            timestamp=candle.close_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            position_size=position.qty,
            direction=position.direction.lower(),
            gross_pnl=gross,
            fees=fees,
            net_pnl=net,
            balance=balance,
            exit_reason=reason,
        ),
        balance,
    )


def run_backtest(session: requests.Session, cfg: Config) -> None:
    candles = fetch_klines(session, cfg)
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]

    ema20 = ema(closes, cfg.ema_fast)
    ema50 = ema(closes, cfg.ema_trend)
    rsi14 = rsi(closes, cfg.rsi_period)
    atr14 = atr(candles, cfg.atr_period)
    vol_sma20 = sma(volumes, cfg.volume_ma_period)

    warmup = max(cfg.ema_trend, cfg.rsi_period, cfg.atr_period, cfg.volume_ma_period) + 1

    balance = cfg.initial_balance
    equity_curve = [balance]
    trades: list[Trade] = []
    open_positions: list[Position] = []

    spread_cache: float | None = None

    for i in range(warmup, len(candles)):
        c = candles[i]
        prev = candles[i - 1]

        # Spread check from futures book ticker
        try:
            spread_cache = fetch_spread(session, cfg.symbol)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Spread fetch failed at i=%d: %s", i, exc)
            if spread_cache is None:
                continue

        spread = spread_cache
        e20 = ema20[i]
        e50 = ema50[i]
        r = rsi14[i]
        a = atr14[i]
        vma = vol_sma20[i]

        if e20 is None or e50 is None or r is None or a is None or vma is None:
            continue

        # Manage existing open positions (trailing + exits)
        still_open: list[Position] = []
        for p in open_positions:
            if cfg.use_trailing_stop:
                if p.direction == "LONG":
                    p.stop_loss = max(p.stop_loss, c.close - (cfg.trailing_atr_mult * a))
                else:
                    p.stop_loss = min(p.stop_loss, c.close + (cfg.trailing_atr_mult * a))

            trade, balance = evaluate_exit(p, c, cfg.fee_rate, balance)
            if trade is None:
                still_open.append(p)
                continue

            trades.append(trade)
            equity_curve.append(balance)
            logging.info(
                "Closed %s entry=%.4f exit=%.4f gross=%.4f fees=%.4f net=%.4f bal=%.4f",
                trade.direction,
                trade.entry_price,
                trade.exit_price,
                trade.gross_pnl,
                trade.fees,
                trade.net_pnl,
                trade.balance,
            )

            if cfg.send_telegram:
                try:
                    send_telegram_message(
                        session,
                        (
                            "Trade Closed\n"
                            f"Symbol: {cfg.symbol}\n"
                            f"Direction: {trade.direction}\n"
                            f"Entry: {trade.entry_price:.4f}\n"
                            f"Exit: {trade.exit_price:.4f}\n"
                            f"Net P/L: {trade.net_pnl:.4f} USD\n"
                            f"Balance: {trade.balance:.4f} USD"
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    logging.error("Telegram trade message failed: %s", exc)

        open_positions = still_open

        # Filters
        volume_ok = c.volume > vma
        spread_ok = spread <= cfg.max_spread
        logging.info(
            "Filter check i=%d volume_ok=%s spread_ok=%s vol=%.6f vma=%.6f spread=%.6f",
            i,
            volume_ok,
            spread_ok,
            c.volume,
            vma,
            spread,
        )

        if len(open_positions) >= cfg.max_concurrent_positions:
            continue
        if not volume_ok or not spread_ok:
            continue

        # Trend Pullback entries (preserved)
        long_trend = c.close > e50
        short_trend = c.close < e50
        prev_e20 = ema20[i - 1] if ema20[i - 1] is not None else e20
        long_trigger = prev.close <= prev_e20 and c.close > e20
        short_trigger = prev.close >= prev_e20 and c.close < e20

        long_rsi_ok = (r > cfg.rsi_threshold) if cfg.use_rsi_filter else True
        short_rsi_ok = (r < cfg.rsi_threshold) if cfg.use_rsi_filter else True

        direction: str | None = None
        if long_trend and long_trigger and long_rsi_ok:
            direction = "LONG"
        elif short_trend and short_trigger and short_rsi_ok:
            direction = "SHORT"

        if direction is None:
            continue

        risk = cfg.atr_sl_mult * a
        if risk <= 0:
            continue

        entry = c.close
        # Fixed 90 USD per trade (or whatever config says), capped by available balance.
        notional = min(cfg.trade_size_usd, balance)
        qty = notional / entry if entry > 0 else 0.0
        if qty <= 0:
            continue

        if direction == "LONG":
            stop = entry - risk
            take = entry + (cfg.rr_ratio * risk)
        else:
            stop = entry + risk
            take = entry - (cfg.rr_ratio * risk)

        open_positions.append(
            Position(
                direction=direction,
                entry_time=c.close_time,
                entry_price=entry,
                qty=qty,
                stop_loss=stop,
                take_profit=take,
            )
        )
        logging.info("Opened %s entry=%.4f qty=%.6f sl=%.4f tp=%.4f", direction, entry, qty, stop, take)

    write_trades_csv(cfg.output_csv, trades)

    # Summary metrics
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
    total_net = sum(t.net_pnl for t in trades)
    avg_return = (total_net / total_trades) if total_trades else 0.0
    max_dd = max_drawdown(equity_curve) * 100.0
    final_balance = balance

    summary = (
        "Backtest Summary\n"
        f"Symbol: {cfg.symbol}\n"
        f"Interval: {cfg.interval}\n"
        f"Total Trades: {total_trades}\n"
        f"Total Net Profit/Loss: {total_net:.4f} USD\n"
        f"Win Rate: {win_rate:.2f}%\n"
        f"Average Return/Trade: {avg_return:.4f} USD\n"
        f"Max Drawdown: {max_dd:.2f}%\n"
        f"Final Account Balance: {final_balance:.4f} USD\n"
        f"CSV: {cfg.output_csv}"
    )

    print("\n=== Backtest Summary ===")
    print(summary)

    if cfg.send_telegram:
        try:
            send_telegram_message(session, summary)
        except Exception as exc:  # noqa: BLE001
            logging.error("Telegram summary failed: %s", exc)


# -----------------------------
# CLI entrypoint
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Binance Futures Trend Pullback backtest + Telegram")
    parser.add_argument("--config", type=str, help="Path to JSON config")
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--interval", type=str)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-time-ms", type=int)
    parser.add_argument("--end-time-ms", type=int)
    parser.add_argument("--max-spread", type=float)
    parser.add_argument("--fee-rate", type=float)
    parser.add_argument("--trade-size-usd", type=float)
    parser.add_argument("--output-csv", type=str)
    parser.add_argument("--max-concurrent-positions", type=int)
    parser.add_argument("--no-telegram", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = load_env_config()
    if args.config:
        cfg = merge_config(cfg, load_json_config(args.config))

    if args.symbol:
        cfg.symbol = args.symbol
    if args.interval:
        cfg.interval = args.interval
    if args.limit is not None:
        cfg.limit = args.limit
    if args.start_time_ms is not None:
        cfg.start_time_ms = args.start_time_ms
    if args.end_time_ms is not None:
        cfg.end_time_ms = args.end_time_ms
    if args.max_spread is not None:
        cfg.max_spread = args.max_spread
    if args.fee_rate is not None:
        cfg.fee_rate = args.fee_rate
    if args.trade_size_usd is not None:
        cfg.trade_size_usd = args.trade_size_usd
    if args.output_csv:
        cfg.output_csv = args.output_csv
    if args.max_concurrent_positions is not None:
        cfg.max_concurrent_positions = args.max_concurrent_positions
    if args.no_telegram:
        cfg.send_telegram = False

    validate_config(cfg)
    return cfg


def main() -> int:
    setup_logging()
    maybe_load_dotenv()
    try:
        cfg = build_config(parse_args())
        logging.info(
            "Starting paper backtest symbol=%s interval=%s limit=%d fee=%.6f trade_size=%.2f",
            cfg.symbol,
            cfg.interval,
            cfg.limit,
            cfg.fee_rate,
            cfg.trade_size_usd,
        )
        with requests.Session() as session:
            run_backtest(session, cfg)
        return 0
    except Exception as exc:  # noqa: BLE001
        logging.error("Script failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
