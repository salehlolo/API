#!/usr/bin/env python3
"""
Unified Binance Futures Trend Pullback Backtester + Telegram Reporter
====================================================================

This script consolidates prior BTC/Telegram and backtesting utilities into one
production-oriented, historical backtesting tool (no live order placement).

Core features
-------------
- Fetch Binance Futures klines from: GET /fapi/v1/klines
- Fetch Binance Futures spread from: GET /fapi/v1/ticker/bookTicker
- Strategy: Trend Pullback with filters
  * Volume filter (mandatory): volume > volume_sma(period)
  * Spread filter (mandatory): spread <= max_spread
  * RSI filter (optional): rsi > threshold for long, rsi < threshold for short
- Entry logic
  * Long: close > EMA(trend) and pullback reclaim over EMA(fast)
  * Short: close < EMA(trend) and pullback rejection below EMA(fast)
- Exit logic
  * SL = ATR * atr_sl_mult
  * TP = ATR * rr_ratio * atr_sl_mult
  * Optional ATR trailing stop
- Risk model
  * Initial balance default: 100 USD
  * Position sizing default: 90% of available balance
  * Fee model default: 0.04% per side (0.0004)
  * Supports max concurrent open positions
- Reporting
  * Per-trade Telegram messages (optional)
  * End-of-backtest summary to console + Telegram (optional)
  * CSV trade log output with requested columns

Environment variables
---------------------
Required for Telegram mode:
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID

Optional strategy/runtime overrides:
- SYMBOL=BTCUSDT
- INTERVAL=1h
- LIMIT=500
- START_TIME_MS=
- END_TIME_MS=
- INITIAL_BALANCE=100
- POSITION_SIZE_PCT=0.9
- FEE_RATE=0.0004
- MAX_CONCURRENT_POSITIONS=1
- EMA_FAST=20
- EMA_TREND=50
- RSI_PERIOD=14
- ATR_PERIOD=14
- VOLUME_MA_PERIOD=20
- USE_RSI_FILTER=true
- RSI_THRESHOLD=50
- VOLUME_MIN_MULTIPLIER=1.0
- MAX_SPREAD=2.0
- ATR_SL_MULT=1.0
- RR_RATIO=1.5
- USE_TRAILING_STOP=false
- TRAILING_ATR_MULT=1.0
- OUTPUT_CSV=backtest_results.csv
- SEND_TELEGRAM=true

CLI examples
------------
- python futures_trend_pullback_telegram.py --no-telegram
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

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

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
    max_concurrent_positions: int = 1

    ema_fast: int = 20
    ema_trend: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    volume_ma_period: int = 20

    use_rsi_filter: bool = True
    rsi_threshold: float = 50.0
    max_spread: float = 2.0
    volume_min_multiplier: float = 1.0

    atr_sl_mult: float = 1.0
    rr_ratio: float = 1.5
    use_trailing_stop: bool = False
    trailing_atr_mult: float = 1.0

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
    reason: str


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
    loaded = load_dotenv()
    if loaded:
        logging.info("Loaded environment variables from .env")


def parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw and raw.strip() else default


def parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw and raw.strip() else default


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
                sleep_backoff(attempt)
                continue
            return response
        except (Timeout, RequestException) as exc:
            last_exc = exc
            logging.warning("Request failed for %s (attempt %d/%d): %s", url, attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                break
            sleep_backoff(attempt)

    raise RuntimeError(f"Request failed after {MAX_RETRIES} attempts: {url}") from last_exc


def sleep_backoff(attempt: int) -> None:
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
        params: dict[str, Any] = {"symbol": cfg.symbol, "interval": cfg.interval, "limit": chunk}
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

        parsed = [
            Candle(
                open_time=int(k[0]),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                close_time=int(k[6]),
            )
            for k in payload
        ]
        candles.extend(parsed)
        remaining = cfg.limit - len(candles)
        if len(parsed) < chunk:
            break
        start_time = parsed[-1].close_time + 1

    candles = candles[: cfg.limit]
    if not candles:
        raise RuntimeError("No futures candles returned")
    logging.info("Fetched %d candles for %s %s", len(candles), cfg.symbol, cfg.interval)
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
    out: list[float | None] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return out
    alpha = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = alpha * values[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def sma(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return out
    rolling = sum(values[:period])
    out[period - 1] = rolling / period
    for i in range(period, len(values)):
        rolling += values[i] - values[i - period]
        out[i] = rolling / period
    return out


def rsi(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if period <= 0 or len(values) <= period:
        return out

    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    def compute(g: float, l: float) -> float:
        if l == 0:
            return 100.0
        rs = g / l
        return 100.0 - (100.0 / (1.0 + rs))

    out[period] = compute(avg_gain, avg_loss)
    for i in range(period + 1, len(values)):
        diff = values[i] - values[i - 1]
        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        out[i] = compute(avg_gain, avg_loss)
    return out


def atr(candles: list[Candle], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(candles)
    if period <= 0 or len(candles) <= period:
        return out

    trs = []
    for i, c in enumerate(candles):
        if i == 0:
            tr = c.high - c.low
        else:
            prev_close = candles[i - 1].close
            tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
        trs.append(tr)

    seed = sum(trs[1 : period + 1]) / period
    out[period] = seed
    prev_atr = seed
    for i in range(period + 1, len(candles)):
        prev_atr = ((prev_atr * (period - 1)) + trs[i]) / period
        out[i] = prev_atr
    return out


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
    body = response.json()
    if not isinstance(body, dict) or body.get("ok") is not True:
        raise RuntimeError(f"Telegram API error payload: {body}")


def timestamp_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0.0
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
            "exit_reason",
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
                t.reason,
            ])


def load_config_from_env() -> Config:
    return Config(
        symbol=os.getenv("SYMBOL", "BTCUSDT"),
        interval=os.getenv("INTERVAL", "1h"),
        limit=parse_int_env("LIMIT", 500),
        start_time_ms=int(os.getenv("START_TIME_MS")) if os.getenv("START_TIME_MS") else None,
        end_time_ms=int(os.getenv("END_TIME_MS")) if os.getenv("END_TIME_MS") else None,
        initial_balance=parse_float_env("INITIAL_BALANCE", 100.0),
        position_size_pct=parse_float_env("POSITION_SIZE_PCT", 0.9),
        fee_rate=parse_float_env("FEE_RATE", 0.0004),
        max_concurrent_positions=parse_int_env("MAX_CONCURRENT_POSITIONS", 1),
        ema_fast=parse_int_env("EMA_FAST", 20),
        ema_trend=parse_int_env("EMA_TREND", 50),
        rsi_period=parse_int_env("RSI_PERIOD", 14),
        atr_period=parse_int_env("ATR_PERIOD", 14),
        volume_ma_period=parse_int_env("VOLUME_MA_PERIOD", 20),
        use_rsi_filter=parse_bool(os.getenv("USE_RSI_FILTER"), True),
        rsi_threshold=parse_float_env("RSI_THRESHOLD", 50.0),
        max_spread=parse_float_env("MAX_SPREAD", 2.0),
        volume_min_multiplier=parse_float_env("VOLUME_MIN_MULTIPLIER", 1.0),
        atr_sl_mult=parse_float_env("ATR_SL_MULT", 1.0),
        rr_ratio=parse_float_env("RR_RATIO", 1.5),
        use_trailing_stop=parse_bool(os.getenv("USE_TRAILING_STOP"), False),
        trailing_atr_mult=parse_float_env("TRAILING_ATR_MULT", 1.0),
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
    if cfg.max_concurrent_positions <= 0:
        raise ValueError("max_concurrent_positions must be > 0")
    if cfg.ema_fast <= 1 or cfg.ema_trend <= 1 or cfg.ema_fast >= cfg.ema_trend:
        raise ValueError("EMA settings invalid")
    if cfg.rsi_period <= 1 or cfg.atr_period <= 1 or cfg.volume_ma_period <= 1:
        raise ValueError("Indicator periods must be > 1")
    if cfg.max_spread < 0:
        raise ValueError("max_spread must be >= 0")
    if cfg.volume_min_multiplier <= 0:
        raise ValueError("volume_min_multiplier must be > 0")
    if cfg.atr_sl_mult <= 0 or cfg.rr_ratio <= 0:
        raise ValueError("atr_sl_mult and rr_ratio must be > 0")
    if cfg.use_trailing_stop and cfg.trailing_atr_mult <= 0:
        raise ValueError("trailing_atr_mult must be > 0 when trailing stop is enabled")


def process_position_exit(
    position: Position,
    candle: Candle,
    fee_rate: float,
    balance: float,
) -> tuple[Trade | None, float]:
    exit_price = None
    reason = ""

    if position.direction == "LONG":
        sl_hit = candle.low <= position.stop_loss
        tp_hit = candle.high >= position.take_profit
        if sl_hit and tp_hit:
            exit_price = position.stop_loss
            reason = "SL_and_TP_same_candle_assume_SL"
        elif sl_hit:
            exit_price = position.stop_loss
            reason = "SL"
        elif tp_hit:
            exit_price = position.take_profit
            reason = "TP"
    else:
        sl_hit = candle.high >= position.stop_loss
        tp_hit = candle.low <= position.take_profit
        if sl_hit and tp_hit:
            exit_price = position.stop_loss
            reason = "SL_and_TP_same_candle_assume_SL"
        elif sl_hit:
            exit_price = position.stop_loss
            reason = "SL"
        elif tp_hit:
            exit_price = position.take_profit
            reason = "TP"

    if exit_price is None:
        return None, balance

    entry_notional = position.entry_price * position.qty
    exit_notional = exit_price * position.qty
    fee_entry = entry_notional * fee_rate
    fee_exit = exit_notional * fee_rate

    gross_pnl = (
        (exit_price - position.entry_price) * position.qty
        if position.direction == "LONG"
        else (position.entry_price - exit_price) * position.qty
    )
    net_pnl = gross_pnl - fee_entry - fee_exit
    balance += net_pnl

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
        reason=reason,
    )
    return trade, balance


def run_backtest(session: requests.Session, cfg: Config) -> None:
    candles = fetch_klines(session, cfg)
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]

    ema_fast = ema(closes, cfg.ema_fast)
    ema_trend = ema(closes, cfg.ema_trend)
    rsi_values = rsi(closes, cfg.rsi_period)
    atr_values = atr(candles, cfg.atr_period)
    volume_sma = sma(volumes, cfg.volume_ma_period)

    warmup = max(cfg.ema_trend, cfg.rsi_period, cfg.atr_period, cfg.volume_ma_period) + 1

    balance = cfg.initial_balance
    equity_curve = [balance]
    trades: list[Trade] = []
    open_positions: list[Position] = []

    spread_cache: float | None = None

    for i in range(warmup, len(candles)):
        candle = candles[i]
        prev_candle = candles[i - 1]

        try:
            spread_cache = fetch_spread(session, cfg.symbol) if spread_cache is not None else fetch_spread(session, cfg.symbol)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Spread fetch failed at candle %d, using cache: %s", i, exc)
            if spread_cache is None:
                continue

        spread = spread_cache
        e20 = ema_fast[i]
        e50 = ema_trend[i]
        r = rsi_values[i]
        a = atr_values[i]
        vma = volume_sma[i]

        if e20 is None or e50 is None or r is None or a is None or vma is None:
            continue

        # Update trailing stops + evaluate exits for all open positions.
        still_open: list[Position] = []
        for pos in open_positions:
            if cfg.use_trailing_stop:
                if pos.direction == "LONG":
                    trail = candle.close - (cfg.trailing_atr_mult * a)
                    pos.stop_loss = max(pos.stop_loss, trail)
                else:
                    trail = candle.close + (cfg.trailing_atr_mult * a)
                    pos.stop_loss = min(pos.stop_loss, trail)

            closed_trade, balance = process_position_exit(pos, candle, cfg.fee_rate, balance)
            if closed_trade is None:
                still_open.append(pos)
                continue

            trades.append(closed_trade)
            equity_curve.append(balance)
            logging.info(
                "Trade closed | dir=%s entry=%.2f exit=%.2f pnl=%.4f balance=%.4f reason=%s",
                closed_trade.direction,
                closed_trade.entry_price,
                closed_trade.exit_price,
                closed_trade.pnl,
                closed_trade.balance,
                closed_trade.reason,
            )

            if cfg.send_telegram:
                message = (
                    "Trade Closed\n"
                    f"Symbol: {cfg.symbol}\n"
                    f"Direction: {closed_trade.direction}\n"
                    f"Entry: {closed_trade.entry_price:.4f}\n"
                    f"Exit: {closed_trade.exit_price:.4f}\n"
                    f"Qty: {closed_trade.position_size:.6f}\n"
                    f"PnL: {closed_trade.pnl:.4f} USD\n"
                    f"Balance: {closed_trade.balance:.4f} USD\n"
                    f"Reason: {closed_trade.reason}"
                )
                try:
                    send_telegram_message(session, message)
                except Exception as exc:  # noqa: BLE001
                    logging.error("Telegram trade message failed: %s", exc)

        open_positions = still_open

        # Entry filters
        if len(open_positions) >= cfg.max_concurrent_positions:
            continue
        if not (candle.volume > (vma * cfg.volume_min_multiplier)):
            continue
        if spread > cfg.max_spread:
            continue

        long_trend = candle.close > e50
        short_trend = candle.close < e50
        prev_e20 = ema_fast[i - 1] if ema_fast[i - 1] is not None else e20

        long_trigger = prev_candle.close <= prev_e20 and candle.close > e20
        short_trigger = prev_candle.close >= prev_e20 and candle.close < e20

        if cfg.use_rsi_filter:
            long_rsi_ok = r > cfg.rsi_threshold
            short_rsi_ok = r < cfg.rsi_threshold
        else:
            long_rsi_ok = True
            short_rsi_ok = True

        direction: str | None = None
        if long_trend and long_trigger and long_rsi_ok:
            direction = "LONG"
        elif short_trend and short_trigger and short_rsi_ok:
            direction = "SHORT"

        if direction is None:
            continue

        risk_per_unit = cfg.atr_sl_mult * a
        if risk_per_unit <= 0:
            continue

        entry = candle.close
        capital_alloc = balance * cfg.position_size_pct
        qty = capital_alloc / entry if entry > 0 else 0.0
        if qty <= 0:
            continue

        if direction == "LONG":
            stop = entry - risk_per_unit
            take = entry + (cfg.rr_ratio * risk_per_unit)
        else:
            stop = entry + risk_per_unit
            take = entry - (cfg.rr_ratio * risk_per_unit)

        open_positions.append(
            Position(
                direction=direction,
                entry_time=candle.close_time,
                entry_price=entry,
                qty=qty,
                stop_loss=stop,
                take_profit=take,
                risk_per_unit=risk_per_unit,
            )
        )
        logging.info(
            "Opened %s | price=%.2f qty=%.6f sl=%.2f tp=%.2f open_positions=%d",
            direction,
            entry,
            qty,
            stop,
            take,
            len(open_positions),
        )

    save_trades_csv(cfg.output_csv, trades)

    total = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = (wins / total * 100.0) if total else 0.0
    avg_pnl = sum(t.pnl for t in trades) / total if total else 0.0
    net_profit = balance - cfg.initial_balance
    total_profit_loss = sum(t.pnl for t in trades)
    drawdown_pct = max_drawdown(equity_curve) * 100.0

    summary = (
        "Backtest Summary\n"
        f"Symbol: {cfg.symbol}\n"
        f"Interval: {cfg.interval}\n"
        f"Trades: {total}\n"
        f"Win Rate: {win_rate:.2f}%\n"
        f"Average P/L per trade: {avg_pnl:.4f} USD\n"
        f"Total Profit/Loss: {total_profit_loss:.4f} USD\n"
        f"Net Profit: {net_profit:.4f} USD\n"
        f"Max Drawdown: {drawdown_pct:.2f}%\n"
        f"Final Balance: {balance:.4f} USD\n"
        f"CSV: {cfg.output_csv}"
    )

    print("\n=== Backtest Summary ===")
    print(summary)

    if cfg.send_telegram:
        try:
            send_telegram_message(session, summary)
        except Exception as exc:  # noqa: BLE001
            logging.error("Telegram summary message failed: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Binance Futures Trend Pullback backtest + Telegram")
    parser.add_argument("--config", type=str, help="Path to JSON config")
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--interval", type=str)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-time-ms", type=int)
    parser.add_argument("--end-time-ms", type=int)
    parser.add_argument("--output-csv", type=str)
    parser.add_argument("--max-concurrent-positions", type=int)
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
    if args.limit is not None:
        cfg.limit = args.limit
    if args.start_time_ms is not None:
        cfg.start_time_ms = args.start_time_ms
    if args.end_time_ms is not None:
        cfg.end_time_ms = args.end_time_ms
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
        args = parse_args()
        cfg = build_config(args)
        logging.info(
            "Starting backtest | symbol=%s interval=%s limit=%d max_open=%d",
            cfg.symbol,
            cfg.interval,
            cfg.limit,
            cfg.max_concurrent_positions,
        )

        with requests.Session() as session:
            run_backtest(session, cfg)
        return 0
    except Exception as exc:  # noqa: BLE001
        logging.error("Script failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
