#!/usr/bin/env python3
"""
Trend Pullback Backtester (Binance Spot)
=======================================

Production-oriented Python 3.11 backtesting script for a Trend Pullback strategy.

Features
--------
- Fetches historical klines from Binance Spot REST API.
- Computes EMA(20), EMA(50), RSI(14), ATR(14), and Volume SMA(20).
- Applies mandatory filters:
  - volume > volume_sma
  - spread <= max_spread (spread from /api/v3/ticker/bookTicker)
- Optional RSI regime filter:
  - long requires RSI > 50
  - short requires RSI < 50
- Entry logic (Trend Pullback):
  - Long: price above EMA50, previous close <= EMA20, current close > EMA20
  - Short: price below EMA50, previous close >= EMA20, current close < EMA20
- Exit logic:
  - stop-loss = 1x ATR, take-profit = 1.5x ATR (configurable risk_reward)
  - optional trailing stop by ATR multiple
- Reports metrics:
  - number of trades, win rate, average R multiple, max drawdown, net profit
- Optional CSV export of trade logs.

No live orders are placed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests
from requests import Response
from requests.exceptions import RequestException, Timeout

BINANCE_BASE = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"
BOOK_TICKER_ENDPOINT = "/api/v3/ticker/bookTicker"

TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}
REQUEST_TIMEOUT = (5, 15)  # connect, read
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 20.0
JITTER_MAX = 0.5


@dataclass
class Config:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    limit: int = 500
    start_time_ms: int | None = None
    end_time_ms: int | None = None

    ema_fast: int = 20
    ema_trend: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    volume_ma_period: int = 20

    max_spread: float = 2.0
    use_rsi_filter: bool = False

    risk_reward: float = 1.5
    atr_stop_mult: float = 1.0
    trailing_stop: bool = False
    trailing_atr_mult: float = 1.0

    initial_balance: float = 10000.0
    position_size_pct: float = 1.0
    fee_rate: float = 0.001

    use_live_spread_per_tick: bool = True
    spread_refresh_seconds: int = 5

    output_csv: str | None = None


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
    initial_risk_per_unit: float


@dataclass
class Trade:
    entry_time: int
    exit_time: int
    direction: str
    entry_price: float
    exit_price: float
    qty: float
    stop_loss: float
    take_profit: float
    pnl: float
    rr: float
    exit_reason: str


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
) -> Response:
    last_err: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.request(method=method, url=url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code in TRANSIENT_HTTP_CODES:
                logging.warning(
                    "Transient HTTP %s for %s (attempt %d/%d)",
                    resp.status_code,
                    url,
                    attempt,
                    MAX_RETRIES,
                )
                if attempt == MAX_RETRIES:
                    return resp
                backoff_sleep(attempt)
                continue
            return resp
        except (Timeout, RequestException) as exc:
            last_err = exc
            logging.warning(
                "Request error for %s (attempt %d/%d): %s",
                url,
                attempt,
                MAX_RETRIES,
                exc,
            )
            if attempt == MAX_RETRIES:
                break
            backoff_sleep(attempt)

    raise RuntimeError(f"Request failed after {MAX_RETRIES} attempts: {url}") from last_err


def backoff_sleep(attempt: int) -> None:
    wait_s = min(INITIAL_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF) + random.uniform(0, JITTER_MAX)
    logging.info("Retrying in %.2f seconds", wait_s)
    time.sleep(wait_s)


def parse_kline(raw: list[Any]) -> Candle:
    return Candle(
        open_time=int(raw[0]),
        open=float(raw[1]),
        high=float(raw[2]),
        low=float(raw[3]),
        close=float(raw[4]),
        volume=float(raw[5]),
        close_time=int(raw[6]),
    )


def fetch_klines(session: requests.Session, cfg: Config) -> list[Candle]:
    url = BINANCE_BASE + KLINES_ENDPOINT
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

        resp = request_with_retry(session, "GET", url, params=params)
        if resp.status_code != 200:
            raise RuntimeError(f"Klines fetch failed: HTTP {resp.status_code} - {resp.text}")

        try:
            payload = resp.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError("Klines endpoint returned non-JSON") from exc

        if not isinstance(payload, list):
            raise RuntimeError("Klines payload is not a list")

        if not payload:
            break

        parsed = [parse_kline(row) for row in payload]
        candles.extend(parsed)

        remaining = cfg.limit - len(candles)
        if len(parsed) < chunk:
            break

        start_time = parsed[-1].close_time + 1

    candles = candles[: cfg.limit]
    if not candles:
        raise RuntimeError("No candles fetched from Binance.")

    logging.info("Fetched %d candles for %s (%s)", len(candles), cfg.symbol, cfg.interval)
    return candles


def fetch_spread(session: requests.Session, symbol: str) -> float:
    url = BINANCE_BASE + BOOK_TICKER_ENDPOINT
    resp = request_with_retry(session, "GET", url, params={"symbol": symbol})
    if resp.status_code != 200:
        raise RuntimeError(f"BookTicker failed: HTTP {resp.status_code} - {resp.text}")

    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError("BookTicker payload invalid")

    try:
        bid = float(payload["bidPrice"])
        ask = float(payload["askPrice"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"BookTicker missing/invalid bidPrice/askPrice: {payload}") from exc

    spread = ask - bid
    if spread < 0:
        raise RuntimeError(f"Invalid spread (negative): bid={bid}, ask={ask}")
    return spread


def ema(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return out

    alpha = 2.0 / (period + 1.0)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = alpha * values[i] + (1.0 - alpha) * prev
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

    def calc_rs_i(g: float, l: float) -> float:
        if l == 0:
            return 100.0
        rs = g / l
        return 100.0 - (100.0 / (1.0 + rs))

    out[period] = calc_rs_i(avg_gain, avg_loss)

    for i in range(period + 1, len(values)):
        diff = values[i] - values[i - 1]
        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        out[i] = calc_rs_i(avg_gain, avg_loss)

    return out


def atr(candles: list[Candle], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(candles)
    if period <= 0 or len(candles) <= period:
        return out

    trs: list[float] = []
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


def unix_ms_to_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def calculate_max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def export_trades_csv(path: str, trades: list[Trade]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "entry_time",
                "exit_time",
                "direction",
                "entry_price",
                "exit_price",
                "qty",
                "stop_loss",
                "take_profit",
                "pnl",
                "rr",
                "exit_reason",
            ]
        )
        for t in trades:
            writer.writerow(
                [
                    unix_ms_to_str(t.entry_time),
                    unix_ms_to_str(t.exit_time),
                    t.direction,
                    f"{t.entry_price:.8f}",
                    f"{t.exit_price:.8f}",
                    f"{t.qty:.8f}",
                    f"{t.stop_loss:.8f}",
                    f"{t.take_profit:.8f}",
                    f"{t.pnl:.8f}",
                    f"{t.rr:.4f}",
                    t.exit_reason,
                ]
            )


def validate_cfg(cfg: Config) -> None:
    if cfg.limit < 120:
        raise ValueError("limit must be >= 120 to allow indicators warmup")
    if cfg.ema_fast <= 1 or cfg.ema_trend <= 1:
        raise ValueError("EMA periods must be > 1")
    if cfg.ema_fast >= cfg.ema_trend:
        raise ValueError("ema_fast must be less than ema_trend")
    if cfg.rsi_period <= 1 or cfg.atr_period <= 1 or cfg.volume_ma_period <= 1:
        raise ValueError("rsi_period/atr_period/volume_ma_period must be > 1")
    if cfg.max_spread < 0:
        raise ValueError("max_spread must be >= 0")
    if cfg.risk_reward <= 0 or cfg.atr_stop_mult <= 0:
        raise ValueError("risk_reward and atr_stop_mult must be > 0")
    if cfg.trailing_stop and cfg.trailing_atr_mult <= 0:
        raise ValueError("trailing_atr_mult must be > 0 when trailing_stop=true")
    if cfg.initial_balance <= 0:
        raise ValueError("initial_balance must be > 0")
    if not (0 < cfg.position_size_pct <= 1):
        raise ValueError("position_size_pct must be within (0,1]")
    if cfg.fee_rate < 0:
        raise ValueError("fee_rate must be >= 0")


def merge_config(base: Config, updates: dict[str, Any]) -> Config:
    allowed = set(Config.__dataclass_fields__.keys())
    for k in updates:
        if k not in allowed:
            raise ValueError(f"Unknown config key in JSON: {k}")
    merged = Config(**{**base.__dict__, **updates})
    return merged


def load_json_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON config must be an object")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trend Pullback backtester on Binance spot data")
    parser.add_argument("--config", type=str, help="Path to JSON config")

    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--start-time-ms", type=int)
    parser.add_argument("--end-time-ms", type=int)

    parser.add_argument("--ema-fast", type=int, default=20)
    parser.add_argument("--ema-trend", type=int, default=50)
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--volume-ma-period", type=int, default=20)

    parser.add_argument("--max-spread", type=float, default=2.0)
    parser.add_argument("--use-rsi-filter", action="store_true")

    parser.add_argument("--risk-reward", type=float, default=1.5)
    parser.add_argument("--atr-stop-mult", type=float, default=1.0)
    parser.add_argument("--trailing-stop", action="store_true")
    parser.add_argument("--trailing-atr-mult", type=float, default=1.0)

    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--position-size-pct", type=float, default=1.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)

    parser.add_argument(
        "--no-live-spread-per-tick",
        action="store_true",
        help="If set, uses cached spread refreshed by --spread-refresh-seconds instead of every candle.",
    )
    parser.add_argument("--spread-refresh-seconds", type=int, default=5)

    parser.add_argument("--output-csv", type=str, help="Optional path to export detailed trades CSV")

    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config(
        symbol=args.symbol,
        interval=args.interval,
        limit=args.limit,
        start_time_ms=args.start_time_ms,
        end_time_ms=args.end_time_ms,
        ema_fast=args.ema_fast,
        ema_trend=args.ema_trend,
        rsi_period=args.rsi_period,
        atr_period=args.atr_period,
        volume_ma_period=args.volume_ma_period,
        max_spread=args.max_spread,
        use_rsi_filter=args.use_rsi_filter,
        risk_reward=args.risk_reward,
        atr_stop_mult=args.atr_stop_mult,
        trailing_stop=args.trailing_stop,
        trailing_atr_mult=args.trailing_atr_mult,
        initial_balance=args.initial_balance,
        position_size_pct=args.position_size_pct,
        fee_rate=args.fee_rate,
        use_live_spread_per_tick=not args.no_live_spread_per_tick,
        spread_refresh_seconds=args.spread_refresh_seconds,
        output_csv=args.output_csv,
    )

    if args.config:
        cfg = merge_config(cfg, load_json_config(args.config))

    validate_cfg(cfg)
    return cfg


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

    cached_spread = 0.0
    last_spread_fetch = 0.0

    for i in range(warmup, len(candles)):
        c = candles[i]
        prev = candles[i - 1]

        # Spread from current order book. In backtest this is a practical approximation.
        now = time.time()
        if cfg.use_live_spread_per_tick or (now - last_spread_fetch) >= cfg.spread_refresh_seconds:
            try:
                cached_spread = fetch_spread(session, cfg.symbol)
                last_spread_fetch = now
            except Exception as exc:  # noqa: BLE001
                logging.warning("Spread fetch failed at candle %d: %s", i, exc)
                # Keep previous spread value if available; otherwise skip this candle.
                if last_spread_fetch == 0.0:
                    continue

        spread = cached_spread
        vol_avg = vol_sma[i]
        e20 = ema20[i]
        e50 = ema50[i]
        r = rsi14[i]
        a = atr14[i]

        if vol_avg is None or e20 is None or e50 is None or r is None or a is None:
            continue

        if position is not None:
            if cfg.trailing_stop:
                if position.direction == "LONG":
                    trail = c.close - (cfg.trailing_atr_mult * a)
                    position.stop_loss = max(position.stop_loss, trail)
                else:
                    trail = c.close + (cfg.trailing_atr_mult * a)
                    position.stop_loss = min(position.stop_loss, trail)

            exit_price: float | None = None
            exit_reason = ""

            if position.direction == "LONG":
                sl_hit = c.low <= position.stop_loss
                tp_hit = c.high >= position.take_profit
                if sl_hit and tp_hit:
                    exit_price = position.stop_loss  # conservative assumption
                    exit_reason = "SL_and_TP_same_candle_assume_SL"
                elif sl_hit:
                    exit_price = position.stop_loss
                    exit_reason = "SL"
                elif tp_hit:
                    exit_price = position.take_profit
                    exit_reason = "TP"
            else:
                sl_hit = c.high >= position.stop_loss
                tp_hit = c.low <= position.take_profit
                if sl_hit and tp_hit:
                    exit_price = position.stop_loss
                    exit_reason = "SL_and_TP_same_candle_assume_SL"
                elif sl_hit:
                    exit_price = position.stop_loss
                    exit_reason = "SL"
                elif tp_hit:
                    exit_price = position.take_profit
                    exit_reason = "TP"

            if exit_price is not None:
                if position.direction == "LONG":
                    gross = (exit_price - position.entry_price) * position.qty
                else:
                    gross = (position.entry_price - exit_price) * position.qty

                fees = cfg.fee_rate * (position.entry_price * position.qty + exit_price * position.qty)
                pnl = gross - fees
                balance += pnl

                rr = pnl / (position.initial_risk_per_unit * position.qty) if position.initial_risk_per_unit > 0 else 0.0
                trades.append(
                    Trade(
                        entry_time=position.entry_time,
                        exit_time=c.close_time,
                        direction=position.direction,
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        qty=position.qty,
                        stop_loss=position.stop_loss,
                        take_profit=position.take_profit,
                        pnl=pnl,
                        rr=rr,
                        exit_reason=exit_reason,
                    )
                )
                equity_curve.append(balance)
                position = None

            continue

        # Filters (mandatory)
        if not (c.volume > vol_avg):
            continue
        if spread > cfg.max_spread:
            continue

        long_trend = c.close > e50
        short_trend = c.close < e50

        # Pullback/crossback trigger using previous close and current close around EMA20
        long_pullback_trigger = prev.close <= (ema20[i - 1] or e20) and c.close > e20
        short_pullback_trigger = prev.close >= (ema20[i - 1] or e20) and c.close < e20

        if cfg.use_rsi_filter:
            long_rsi_ok = r > 50
            short_rsi_ok = r < 50
        else:
            long_rsi_ok = True
            short_rsi_ok = True

        direction: str | None = None
        if long_trend and long_pullback_trigger and long_rsi_ok:
            direction = "LONG"
        elif short_trend and short_pullback_trigger and short_rsi_ok:
            direction = "SHORT"

        if direction is None:
            continue

        risk_per_unit = cfg.atr_stop_mult * a
        if risk_per_unit <= 0:
            continue

        entry = c.close
        if direction == "LONG":
            stop = entry - risk_per_unit
            tp = entry + (cfg.risk_reward * risk_per_unit)
        else:
            stop = entry + risk_per_unit
            tp = entry - (cfg.risk_reward * risk_per_unit)

        capital_alloc = balance * cfg.position_size_pct
        qty = capital_alloc / entry if entry > 0 else 0.0
        if qty <= 0:
            continue

        position = Position(
            direction=direction,
            entry_time=c.close_time,
            entry_price=entry,
            qty=qty,
            stop_loss=stop,
            take_profit=tp,
            initial_risk_per_unit=risk_per_unit,
        )

    # Performance metrics
    n = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = (wins / n * 100.0) if n > 0 else 0.0
    avg_rr = sum(t.rr for t in trades) / n if n > 0 else 0.0
    net_profit = balance - cfg.initial_balance
    max_dd = calculate_max_drawdown(equity_curve) * 100.0

    print("\n=== Trend Pullback Backtest Summary ===")
    print(f"Symbol: {cfg.symbol}")
    print(f"Interval: {cfg.interval}")
    print(f"Candles: {len(candles)}")
    print(f"Trades: {n}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Average R multiple: {avg_rr:.4f}")
    print(f"Max drawdown: {max_dd:.2f}%")
    print(f"Net profit: {net_profit:.2f}")
    print(f"Final balance: {balance:.2f}")

    if cfg.output_csv:
        export_trades_csv(cfg.output_csv, trades)
        print(f"Trades exported to: {cfg.output_csv}")


def main() -> int:
    setup_logging()
    try:
        args = parse_args()
        cfg = build_config_from_args(args)
        logging.info("Starting backtest: symbol=%s interval=%s limit=%d", cfg.symbol, cfg.interval, cfg.limit)

        with requests.Session() as session:
            run_backtest(session, cfg)
        return 0
    except Exception as exc:  # noqa: BLE001
        logging.error("Backtest failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
