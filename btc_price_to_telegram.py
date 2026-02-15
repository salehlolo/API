#!/usr/bin/env python3
"""
README (Quick Start)
====================
This script fetches the real-time BTC/USDT spot price from Binance and sends it to Telegram.
It supports one-shot mode (default), periodic mode (`--interval`), and test mode (`--test`).

Requirements
------------
- Python 3.11+
- `requests`
- Optional: `python-dotenv` (to load variables from a local `.env` file)

Install
-------
1) Create and activate a virtual environment (recommended):
   python3.11 -m venv .venv
   source .venv/bin/activate

2) Install dependencies:
   pip install requests python-dotenv

Environment Variables
---------------------
- TELEGRAM_BOT_TOKEN: Telegram bot token from BotFather
- TELEGRAM_CHAT_ID: Target chat ID

You can export them in your shell:
   export TELEGRAM_BOT_TOKEN="123456:ABCDEF..."
   export TELEGRAM_CHAT_ID="123456789"

Or put them in a `.env` file (if python-dotenv is installed):
   TELEGRAM_BOT_TOKEN=123456:ABCDEF...
   TELEGRAM_CHAT_ID=123456789

How to get TELEGRAM_CHAT_ID
---------------------------
1) Send any message to your bot in Telegram.
2) Call:
   https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/getUpdates
3) Inspect the latest update and extract:
   message.chat.id

Run Examples
------------
- Run once (default):
  python btc_price_to_telegram.py

- Run every 60 seconds:
  python btc_price_to_telegram.py --interval 60

- Test Binance fetch + validation only (no Telegram send):
  python btc_price_to_telegram.py --test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import requests
from requests import Response
from requests.exceptions import RequestException, Timeout
from zoneinfo import ZoneInfo

# Optional .env support; no hard dependency.
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
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


@dataclass(frozen=True)
class PriceData:
    symbol: str
    price: Decimal


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
    """HTTP request with retries for transient network and status failures."""
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
    response = request_with_retry(
        session,
        "GET",
        BINANCE_TIME_URL,
        timeout=REQUEST_TIMEOUT,
    )

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

    for key in ("symbol", "price"):
        if key not in payload:
            raise ValueError(f"Price validation failed: missing '{key}' key")

    symbol = payload["symbol"]
    if symbol != BINANCE_SYMBOL:
        raise ValueError(f"Price validation failed: expected symbol {BINANCE_SYMBOL}, got {symbol!r}")

    raw_price = payload["price"]
    try:
        price = Decimal(str(raw_price))
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"Price validation failed: cannot parse price value {raw_price!r}") from exc

    if price <= 0:
        raise ValueError(f"Price validation failed: price must be > 0, got {price}")

    return PriceData(symbol=symbol, price=price)


def build_message(price_data: PriceData) -> str:
    now_cairo = datetime.now(CAIRO_TZ).strftime("%Y-%m-%d %H:%M:%S")
    return f"BTC/USDT: {price_data.price:.2f}\nTime (Africa/Cairo): {now_cairo}"


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

    response = request_with_retry(
        session,
        "POST",
        url,
        timeout=REQUEST_TIMEOUT,
        json_payload=payload,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Telegram API sendMessage failed: HTTP {response.status_code} | {response.text}")

    result = parse_json_response(response, "Telegram sendMessage endpoint")
    if result.get("ok") is not True:
        raise RuntimeError(f"Telegram API returned error payload: {result}")


def safe_send_error_to_telegram(session: requests.Session, err_msg: str) -> None:
    """Best-effort error reporting to Telegram; never raises."""
    try:
        token, chat_id = get_telegram_config()
    except ValueError as cfg_err:
        logging.error("Cannot send error to Telegram: %s", cfg_err)
        return

    try:
        send_telegram_message(session, token, chat_id, f"[ERROR] BTC price bot failed: {err_msg}")
        logging.info("Error message sent to Telegram.")
    except Exception as exc:  # noqa: BLE001 - final fallback logging path
        logging.error("Failed to send error to Telegram: %s", exc)


def run_once(session: requests.Session, test_mode: bool) -> int:
    try:
        server_time = validate_binance_connectivity(session)
        logging.info("Connectivity check passed. Binance serverTime=%s", server_time)

        price_data = fetch_and_validate_price(session)
        logging.info("Price validated successfully: %s %s", price_data.symbol, price_data.price)

        if test_mode:
            logging.info("--test mode enabled: skipping Telegram send.")
            logging.info("Test message preview:\n%s", build_message(price_data))
            return 0

        token, chat_id = get_telegram_config()
        message = build_message(price_data)
        send_telegram_message(session, token, chat_id, message)
        logging.info("Price message sent to Telegram.")
        return 0

    except Exception as exc:  # noqa: BLE001 - top-level operational boundary
        error_message = str(exc)
        logging.error("Run failed: %s", error_message)
        if not test_mode:
            safe_send_error_to_telegram(session, error_message)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch BTC/USDT price from Binance and send to Telegram.")
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Run every N seconds until interrupted (default: run once)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test Binance fetch + validation only; do not send Telegram messages.",
    )
    args = parser.parse_args()

    if args.interval < 0:
        parser.error("--interval must be >= 0")

    return args


def main() -> int:
    configure_logging()
    maybe_load_dotenv()
    args = parse_args()

    with requests.Session() as session:
        if args.interval == 0:
            return run_once(session, test_mode=args.test)

        logging.info("Starting periodic mode with interval=%s seconds", args.interval)
        exit_code = 0
        try:
            while True:
                exit_code = run_once(session, test_mode=args.test)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logging.info("Interrupted by user. Exiting periodic mode.")
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
