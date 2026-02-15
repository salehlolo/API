#!/usr/bin/env python3
"""
Futures Trend Pullback Backtester + Telegram Alerts

What this script does
---------------------
1) Fetches Binance Futures historical candles from:
   GET /fapi/v1/klines
2) Fetches spread from:
   GET /fapi/v1/ticker/bookTicker
3) Runs Trend Pullback backtest with:
   - Volume filter: volume > SMA(volume, 20)
   - Spread filter: spread <= max_spread
   - RSI filter: RSI(14) > 50 for longs, < 50 for shorts (toggleable)
   - Trend: close above/below EMA(50)
   - Pullback trigger around EMA(20)
   - Exits with ATR(14): SL = 1x ATR, TP = 1.5x ATR
4) Applies Binance futures fee assumptions (default 0.04% per side).
5) Sends Telegram message after each closed trade and at end-of-backtest summary.
6) Exports trade logs to CSV.

Environment variables
---------------------
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
Optional overrides:
- SYMBOL (default BTCUSDT)
- INTERVAL (default 1h)
- LIMIT (default 500)
- INITIAL_BALANCE (default 100)
- POSITION_SIZE_PCT (default 0.9)
- FEE_RATE (default 0.0004)
- MAX_SPREAD (default 2.0)
- USE_RSI_FILTER (default true)
- RSI_PERIOD (default 14)
- VOLUME_MA_PERIOD (default 20)
- EMA_FAST (default 20)
- EMA_TREND (default 50)
- ATR_PERIOD (default 14)
- ATR_SL_MULT (default 1.0)
- RR_RATIO (default 1.5)
- OUTPUT_CSV (default backtest_results.csv)

Run examples
------------
- python futures_trend_pullback_telegram.py
- python futures_trend_pullback_telegram.py --symbol ETHUSDT --interval 15m --limit 1000
- python futures_trend_pullback_telegram.py --config config.json
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

FUTURES_BASE = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"
BOOK_TICKER_ENDPOINT = "/fapi/v1/ticker/bookTicker"

TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
REQUEST_TIMEOUT = (5, 15)
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 20.0
MAX_JITTER = 0.5


@dataclass
class Config:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    limit: int = 500
    start_time_ms: int | None = None
    end_time_ms: int | None = None

    initial_balance: float = 100.0
    position_size_pct: float = 0.9
    fee_rate: float = 0.0004

    ema_fast: int = 20
    ema_trend: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    volume_ma_period: int = 20

    use_rsi_filter: bool = True
    max_spread: float = 2.0
    rsi_threshold: float = 50.0

    atr_sl_mult: float = 1.0
    rr_ratio: float = 1.5

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
    risk_per_unit: float


@dataclass
class Trade:
    timestamp: int
    entry_price: float
    exit_price: float
    position_size: float
    direction: str
    pnl: float
    balance: float
    fee_entry: float
    fee_exit: float


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_bool(raw: str, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None and raw.strip() else default


def parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None and raw.strip() else default


def load_config_from_env() -> Config:
    return Config(
        symbol=os.getenv("SYMBOL", "BTCUSDT"),
        interval=os.getenv("INTERVAL", "1h"),
        limit=parse_int_env("LIMIT", 500),
        initial_balance=parse_float_env("INITIAL_BALANCE", 100.0),
        position_size_pct=parse_float_env("POSITION_SIZE_PCT", 0.9),
        fee_rate=parse_float_env("FEE_RATE", 0.0004),
        ema_fast=parse_int_env("EMA_FAST", 20),
        ema_trend=parse_int_env("EMA_TREND", 50),
        rsi_period=parse_int_env("RSI_PERIOD", 14),
        atr_period=parse_int_env("ATR_PERIOD", 14),
        volume_ma_period=parse_int_env("VOLUME_MA_PERIOD", 20),
        use_rsi_filter=parse_bool(os.getenv("USE_RSI_FILTER"), True),
        max_spread=parse_float_env("MAX_SPREAD", 2.0),
        rsi_threshold=parse_float_env("RSI_THRESHOLD", 50.0),
        atr_sl_mult=parse_float_env("ATR_SL_MULT", 1.0),
        rr_ratio=parse_float_env("RR_RATIO", 1.5),
        output_csv=os.getenv("OUTPUT_CSV", "backtest_results.csv"),
        send_telegram=parse_bool(os.getenv("SEND_TELEGRAM"), True),
    )


def load_json_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON config must be an object")
    return data


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
    if not (0 < cfg.position_size_pct <= 1):
        raise ValueError("position_size_pct must be in (0,1]")
    if cfg.fee_rate < 0:
        raise ValueError("fee_rate must be >= 0")
    if cfg.ema_fast <= 1 or cfg.ema_trend <= 1 or cfg.ema_fast >= cfg.ema_trend:
        raise ValueError("EMA settings invalid")
    if cfg.rsi_period <= 1 or cfg.atr_period <= 1 or cfg.volume_ma_period <= 1:
        raise ValueError("Indicator periods must be > 1")
    if cfg.max_spread < 0:
        raise ValueError("max_spread must be >= 0")
    if cfg.atr_sl_mult <= 0 or cfg.rr_ratio <= 0:
        raise ValueError("atr_sl_mult and rr_ratio must be > 0")


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
            response = session.request(
                method=method,
                url=url,
                params=params,
                json=json_payload,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code in TRANSIENT_STATUSES:
                logging.warning(
                    "Transient HTTP %s for %s (attempt %d/%d)",
                    response.status_code,
                    url,
                    attempt,
                    MAX_RETRIES,
                )
                if attempt == MAX_RETRIES:
                    return response
                backoff_sleep(attempt)
                continue
            return response
        except (Timeout, RequestException) as exc:
            last_exc = exc
            logging.warning("Request failed for %s (attempt %d/%d): %s", url, attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                break
            backoff_sleep(attempt)
    raise RuntimeError(f"Request failed after {MAX_RETRIES} attempts: {url}") from last_exc


def backoff_sleep(attempt: int) -> None:
    delay = min(INITIAL_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF) + random.uniform(0, MAX_JITTER)
    logging.info("Retrying in %.2f seconds", delay)
    time.sleep(delay)


def fetch_klines(session: requests.Session, cfg: Config) -> list[Candle]:
    url = FUTURES_BASE + KLINES_ENDPOINT
    candles: list[Candle] = []

    remaining = cfg.limit
    start_time = cfg.start_time_ms
    while remaining > 0:
        chunk = min(remaining, 1000)
        params: dict[str, Any] = {
            "symbol": cfg.symbol,
            "interval": cfg.interval,
            "limit": chunk,
        }
        if start_time is not None:
            params["startTime"] = start_time
        if cfg.end_time_ms is not None:
            params["endTime"] = cfg.end_time_ms

        response = request_with_retry(session, "GET", url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"Klines request failed: HTTP {response.status_code} - {response.text}")

        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError("Invalid klines response format")
        if not payload:
            break

        parsed_chunk = [
            Candle(
                open_time=int(item[0]),
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
                volume=float(item[5]),
                close_time=int(item[6]),
            )
            for item in payload
        ]
        candles.extend(parsed_chunk)
        remaining = cfg.limit - len(candles)
        if len(parsed_chunk) < chunk:
            break
        start_time = parsed_chunk[-1].close_time + 1

    candles = candles[: cfg.limit]
    if not candles:
        raise RuntimeError("No futures klines returned")
    logging.info("Fetched %d futures candles for %s %s", len(candles), cfg.symbol, cfg.interval)
    return candles


def fetch_spread(session: requests.Session, symbol: str) -> float:
    url = FUTURES_BASE + BOOK_TICKER_ENDPOINT
    response = request_with_retry(session, "GET", url, params={"symbol": symbol})
    if response.status_code != 200:
        raise RuntimeError(f"BookTicker request failed: HTTP {response.status_code} - {response.text}")

    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid bookTicker payload")

    ask = float(payload["askPrice"])
    bid = float(payload["bidPrice"])
    spread = ask - bid
    if spread < 0:
        raise RuntimeError(f"Negative spread from Binance: {spread}")
    return spread


def ema(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return result
    alpha = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    result[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = alpha * values[i] + (1 - alpha) * prev
        result[i] = prev
    return result


def sma(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return result
    rolling_sum = sum(values[:period])
    result[period - 1] = rolling_sum / period
    for i in range(period, len(values)):
        rolling_sum += values[i] - values[i - period]
        result[i] = rolling_sum / period
    return result


def rsi(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if period <= 0 or len(values) <= period:
        return result

    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    def calc_rsi(gain: float, loss: float) -> float:
        if loss == 0:
            return 100.0
        rs = gain / loss
        return 100.0 - (100.0 / (1.0 + rs))

    result[period] = calc_rsi(avg_gain, avg_loss)
    for i in range(period + 1, len(values)):
        diff = values[i] - values[i - 1]
        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        result[i] = calc_rsi(avg_gain, avg_loss)
    return result


def atr(candles: list[Candle], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(candles)
    if period <= 0 or len(candles) <= period:
        return result

    true_ranges: list[float] = []
    for i, candle in enumerate(candles):
        if i == 0:
            tr = candle.high - candle.low
        else:
            prev_close = candles[i - 1].close
            tr = max(candle.high - candle.low, abs(candle.high - prev_close), abs(candle.low - prev_close))
        true_ranges.append(tr)

    seed = sum(true_ranges[1 : period + 1]) / period
    result[period] = seed
    prev_atr = seed
    for i in range(period + 1, len(candles)):
        prev_atr = ((prev_atr * (period - 1)) + true_ranges[i]) / period
        result[i] = prev_atr
    return result


def telegram_credentials() -> tuple[str, str]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
    return token, chat_id


def send_telegram_message(session: requests.Session, text: str) -> None:
    token, chat_id = telegram_credentials()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    response = request_with_retry(session, "POST", url, json_payload=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Telegram sendMessage failed: HTTP {response.status_code} - {response.text}")
    payload_resp = response.json()
    if not isinstance(payload_resp, dict) or payload_resp.get("ok") is not True:
        raise RuntimeError(f"Telegram API error: {payload_resp}")


def timestamp_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def max_drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for value in equity:
        peak = max(peak, value)
        dd = (peak - value) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def save_trades_csv(path: str, trades: list[Trade]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "entry_price",
            "exit_price",
            "position_size",
            "trade_direction",
            "profit_loss",
            "account_balance",
            "fee_entry",
            "fee_exit",
        ])
        for t in trades:
            writer.writerow([
                timestamp_str(t.timestamp),
                f"{t.entry_price:.8f}",
                f"{t.exit_price:.8f}",
                f"{t.position_size:.8f}",
                t.direction,
                f"{t.pnl:.8f}",
                f"{t.balance:.8f}",
                f"{t.fee_entry:.8f}",
                f"{t.fee_exit:.8f}",
            ])


def run_backtest(session: requests.Session, cfg: Config) -> None:
    candles = fetch_klines(session, cfg)
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]

    ema20 = ema(closes, cfg.ema_fast)
    ema50 = ema(closes, cfg.ema_trend)
    rsi14 = rsi(closes, cfg.rsi_period)
    atr14 = atr(candles, cfg.atr_period)
    vol_sma = sma(volumes, cfg.volume_ma_period)

    warmup = max(cfg.ema_trend, cfg.rsi_period, cfg.atr_period, cfg.volume_ma_period) + 1

    balance = cfg.initial_balance
    equity_curve = [balance]
    trades: list[Trade] = []
    position: Position | None = None

    spread_cache = None

    for i in range(warmup, len(candles)):
        candle = candles[i]
        prev_candle = candles[i - 1]

        if spread_cache is None:
            spread_cache = fetch_spread(session, cfg.symbol)
        else:
            try:
                spread_cache = fetch_spread(session, cfg.symbol)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Spread fetch failed, using cached value: %s", exc)

        spread = spread_cache
        e20 = ema20[i]
        e50 = ema50[i]
        r = rsi14[i]
        a = atr14[i]
        vma = vol_sma[i]

        if e20 is None or e50 is None or r is None or a is None or vma is None:
            continue

        if position is not None:
            exit_price = None
            if position.direction == "LONG":
                sl_hit = candle.low <= position.stop_loss
                tp_hit = candle.high >= position.take_profit
                if sl_hit and tp_hit:
                    exit_price = position.stop_loss
                elif sl_hit:
                    exit_price = position.stop_loss
                elif tp_hit:
                    exit_price = position.take_profit
            else:
                sl_hit = candle.high >= position.stop_loss
                tp_hit = candle.low <= position.take_profit
                if sl_hit and tp_hit:
                    exit_price = position.stop_loss
                elif sl_hit:
                    exit_price = position.stop_loss
                elif tp_hit:
                    exit_price = position.take_profit

            if exit_price is not None:
                entry_notional = position.entry_price * position.qty
                exit_notional = exit_price * position.qty
                fee_entry = entry_notional * cfg.fee_rate
                fee_exit = exit_notional * cfg.fee_rate

                gross_pnl = (
                    (exit_price - position.entry_price) * position.qty
                    if position.direction == "LONG"
                    else (position.entry_price - exit_price) * position.qty
                )
                net_pnl = gross_pnl - fee_entry - fee_exit
                balance += net_pnl
                equity_curve.append(balance)

                trade = Trade(
                    timestamp=candle.close_time,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    position_size=position.qty,
                    direction=position.direction.lower(),
                    pnl=net_pnl,
                    balance=balance,
                    fee_entry=fee_entry,
                    fee_exit=fee_exit,
                )
                trades.append(trade)

                if cfg.send_telegram:
                    msg = (
                        "Trade Closed\n"
                        f"Symbol: {cfg.symbol}\n"
                        f"Direction: {trade.direction}\n"
                        f"Entry: {trade.entry_price:.2f}\n"
                        f"Exit: {trade.exit_price:.2f}\n"
                        f"Qty: {trade.position_size:.6f}\n"
                        f"PnL: {trade.pnl:.4f} USD\n"
                        f"Balance: {trade.balance:.4f} USD"
                    )
                    try:
                        send_telegram_message(session, msg)
                    except Exception as exc:  # noqa: BLE001
                        logging.error("Failed sending trade Telegram message: %s", exc)

                position = None
            continue

        if not (candle.volume > vma):
            continue
        if spread > cfg.max_spread:
            continue

        long_trend = candle.close > e50
        short_trend = candle.close < e50
        prev_e20 = ema20[i - 1] if ema20[i - 1] is not None else e20

        long_trigger = prev_candle.close <= prev_e20 and candle.close > e20
        short_trigger = prev_candle.close >= prev_e20 and candle.close < e20

        if cfg.use_rsi_filter:
            long_rsi_ok = r > cfg.rsi_threshold
            short_rsi_ok = r < cfg.rsi_threshold
        else:
            long_rsi_ok = True
            short_rsi_ok = True

        direction = None
        if long_trend and long_trigger and long_rsi_ok:
            direction = "LONG"
        elif short_trend and short_trigger and short_rsi_ok:
            direction = "SHORT"

        if direction is None:
            continue

        risk_per_unit = cfg.atr_sl_mult * a
        if risk_per_unit <= 0:
            continue

        entry_price = candle.close
        capital_to_use = balance * cfg.position_size_pct
        qty = capital_to_use / entry_price if entry_price > 0 else 0
        if qty <= 0:
            continue

        if direction == "LONG":
            stop = entry_price - risk_per_unit
            take = entry_price + (cfg.rr_ratio * risk_per_unit)
        else:
            stop = entry_price + risk_per_unit
            take = entry_price - (cfg.rr_ratio * risk_per_unit)

        position = Position(
            direction=direction,
            entry_time=candle.close_time,
            entry_price=entry_price,
            qty=qty,
            stop_loss=stop,
            take_profit=take,
            risk_per_unit=risk_per_unit,
        )

    wins = sum(1 for t in trades if t.pnl > 0)
    total = len(trades)
    win_rate = (wins / total * 100.0) if total else 0.0
    avg_pnl = sum(t.pnl for t in trades) / total if total else 0.0
    net_profit = balance - cfg.initial_balance
    dd_pct = max_drawdown(equity_curve) * 100

    save_trades_csv(cfg.output_csv, trades)

    summary = (
        "Backtest Summary\n"
        f"Symbol: {cfg.symbol}\n"
        f"Interval: {cfg.interval}\n"
        f"Trades: {total}\n"
        f"Win Rate: {win_rate:.2f}%\n"
        f"Avg PnL/Trade: {avg_pnl:.4f} USD\n"
        f"Max Drawdown: {dd_pct:.2f}%\n"
        f"Net Profit: {net_profit:.4f} USD\n"
        f"Final Balance: {balance:.4f} USD\n"
        f"CSV: {cfg.output_csv}"
    )

    print("\n=== Backtest Summary ===")
    print(summary)

    if cfg.send_telegram:
        try:
            send_telegram_message(session, summary)
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed sending summary Telegram message: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binance Futures Trend Pullback backtest + Telegram")
    parser.add_argument("--config", type=str, help="Path to JSON config")
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--interval", type=str)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-time-ms", type=int)
    parser.add_argument("--end-time-ms", type=int)
    parser.add_argument("--output-csv", type=str)
    parser.add_argument("--no-telegram", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = load_config_from_env()

    if args.config:
        cfg = merge_config(cfg, load_json_config(args.config))

    if args.symbol:
        cfg.symbol = args.symbol
    if args.interval:
        cfg.interval = args.interval
    if args.limit:
        cfg.limit = args.limit
    if args.start_time_ms is not None:
        cfg.start_time_ms = args.start_time_ms
    if args.end_time_ms is not None:
        cfg.end_time_ms = args.end_time_ms
    if args.output_csv:
        cfg.output_csv = args.output_csv
    if args.no_telegram:
        cfg.send_telegram = False

    validate_config(cfg)
    return cfg


def main() -> int:
    setup_logging()
    try:
        args = parse_args()
        cfg = build_config(args)

        logging.info(
            "Starting futures backtest | symbol=%s interval=%s limit=%d",
            cfg.symbol,
            cfg.interval,
            cfg.limit,
        )

        with requests.Session() as session:
            run_backtest(session, cfg)

        return 0
    except Exception as exc:  # noqa: BLE001
        logging.error("Script failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
