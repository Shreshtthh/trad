#!/usr/bin/env python3
"""
Backtest — replay 30 days through the exact same pipeline (no LLM, no MCP, no TWAK).

Usage:
    python3 scripts/backtest.py                      # default: 30 days, 30 tokens
    python3 scripts/backtest.py --days 7 --top 10   # quick: 7 days, 10 tokens
    python3 scripts/backtest.py --interval 1h       # hourly candles (faster)
    python3 scripts/backtest.py --all-tokens        # full 149-token allowlist
    python3 scripts/backtest.py --cache-only        # just download data, don't simulate

Flow:
    1. Download 15-min k-line candles from CMC for N tokens (cached to disk)
    2. For each tick (every interval):
       a. Determine regime from Fear & Greed historical (rule-based, no LLM)
       b. Compute momentum scores locally from OHLCV data
       c. Generate swap plan via portfolio.generate_swap_plan()
       d. Simulate execution at current close price, track PnL
    3. Report: total PnL, Sharpe, max drawdown, win rate, trades/day

Momentum scoring (local, matches MCP composite structure):
    - 24h price change (40%)     — close vs close 96 periods ago
    - 24h volume surge (20%)     — volume / 96-period average
    - RSI 14-period (20%)        — normalized to 0-1
    - EMA50 ratio (20%)          — close / EMA50
    → composite_score = weighted sum, 0-1 range

Price source: close price at each tick from CMC k-line candles.
Slippage: same 0.985 multiplier as live portfolio.py.
BNB gas buffer: $20 preserved, identical to live config.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

# Add agent/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from data.allowlist import COMP_TOKENS, is_eligible, is_stablecoin
from data.cmc_client import kline_points, fear_and_greed_historical, cmc_fetch_kline_prices
from strategy.portfolio import (
    generate_swap_plan, SwapPlan, SwapInstruction,
    SLIPPAGE_MULTIPLIER, BNB_GAS_BUFFER_USD, MAX_TRADES_PER_DAY,
)
from strategy.momentum import MomentumCandidate

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backtest")

# Suppress noisy per-tick log lines from portfolio module during backtest
logging.getLogger("strategy.portfolio").setLevel(logging.WARNING)

# ── Constants ──
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "backtest"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INTERVAL = "15m"
DEFAULT_DAYS = 30
DEFAULT_TOP_N = 30
INITIAL_USD = 10_000.0

# Candle intervals and their ticks-per-day
INTERVAL_MINUTES: dict[str, int] = {
    "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440,
}


# ═══════════════════════════════════════════════════════════════════════════
# Data Fetching
# ═══════════════════════════════════════════════════════════════════════════

def download_klines(
    symbols: list[str],
    interval: str = DEFAULT_INTERVAL,
    days: int = DEFAULT_DAYS,
) -> dict[str, list[dict]]:
    """
    Download k-line candles for each symbol. Cached to disk.

    Returns {symbol_upper: [candle, ...]} where each candle is
    {timestamp, quote: {USD: {open, high, low, close, volume}}}.
    """
    mins = INTERVAL_MINUTES.get(interval, 15)
    candles_needed = (days * 24 * 60) // mins

    cache_path = CACHE_DIR / f"klines_{interval}_{days}d.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            cached_symbols = set(cached.keys())
            requested = {s.upper() for s in symbols}
            if requested.issubset(cached_symbols):
                log.info("Using cached k-line data: %d symbols from %s", len(cached), cache_path)
                return cached
            log.info("Cache partial: %d/%d symbols — refreshing", len(cached_symbols & requested), len(requested))
        except (json.JSONDecodeError, OSError):
            log.warning("K-line cache corrupt — re-downloading")

    all_data: dict[str, list[dict]] = {}
    total = len(symbols)
    t0 = time.monotonic()

    for i, sym in enumerate(symbols):
        try:
            candles = kline_points(sym, interval=interval, count=candles_needed)
            if candles:
                all_data[sym.upper()] = candles
                if (i + 1) % 10 == 0:
                    log.info("  Downloaded %d/%d tokens (%.0fs elapsed)", i + 1, total, time.monotonic() - t0)
        except Exception as exc:
            log.warning("K-line fetch failed for %s: %s", sym, exc)

    log.info("Downloaded k-line data: %d/%d tokens in %.0fs", len(all_data), total, time.monotonic() - t0)

    # Cache to disk
    with open(cache_path, "w") as f:
        json.dump(all_data, f, default=str)

    return all_data


def download_fear_greed(days: int = DEFAULT_DAYS) -> dict[str, int]:
    """
    Fetch daily Fear & Greed values. Cached to disk.
    Returns {date_str: value} — e.g., {"2026-06-20": 52}.
    """
    cache_path = CACHE_DIR / "fear_greed.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            log.info("Using cached F&G data: %d days", len(cached))
            return cached
        except (json.JSONDecodeError, OSError):
            pass

    fg_map: dict[str, int] = {}
    try:
        items = fear_and_greed_historical(limit=days)
        for item in items:
            ts = item.get("timestamp", "")
            date = ts[:10] if ts else ""
            val = item.get("value", 50)
            if date:
                fg_map[date] = int(val)
        log.info("Fetched F&G: %d days", len(fg_map))
    except Exception as exc:
        log.warning("F&G fetch failed: %s — using neutral=50", exc)

    with open(cache_path, "w") as f:
        json.dump(fg_map, f)

    return fg_map


# ═══════════════════════════════════════════════════════════════════════════
# Momentum Scoring (local — no MCP)
# ═══════════════════════════════════════════════════════════════════════════

def rsi(prices: list[float], period: int = 14) -> float:
    """Compute Wilder's RSI from a list of closing prices (most recent last)."""
    if len(prices) < period + 1:
        return 50.0  # neutral

    # First-pass average gain/loss (simple average)
    gains, losses = 0.0, 0.0
    for i in range(len(prices) - period, len(prices)):
        delta = prices[i] - prices[i - 1]
        if delta > 0:
            gains += delta
        else:
            losses -= delta

    avg_gain = gains / period
    avg_loss = losses / period

    # Wilder smoothing: use all previous values
    # (simplified — single pass is good enough for backtest scoring)
    if avg_gain == 0 and avg_loss == 0:
        return 50.0  # flat → neutral
    if avg_loss == 0:
        return 100.0  # pure uptrend → 100
    if avg_gain == 0:
        return 0.0    # pure downtrend → 0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def ema(series: list[float], period: int) -> float:
    """Exponential moving average of the last `period` values."""
    if not series:
        return 0.0
    alpha = 2.0 / (period + 1)
    result = series[0]
    for val in series[1:]:
        result = alpha * val + (1 - alpha) * result
    return result


def compute_momentum(
    symbol: str,
    candles: list[dict],
    tick_index: int,
) -> Optional[MomentumCandidate]:
    """
    Compute a momentum score from candle data at a given tick index.

    Uses candles[:tick_index+1] for OHLCV up to this moment.
    Requires at least 96 periods of lookback for 24h metrics (at 15m).
    """
    if tick_index < 96:
        return None  # not enough data yet

    closes = [c.get("quote", {}).get("USD", {}).get("close", 0) for c in candles[:tick_index + 1]]
    volumes = [c.get("quote", {}).get("USD", {}).get("volume", 0) for c in candles[:tick_index + 1]]

    if not closes or closes[-1] <= 0:
        return None

    current_close = closes[-1]
    close_96_ago = closes[tick_index - 96] if len(closes) > 96 else closes[0]
    if close_96_ago <= 0:
        return None

    # 1. 24h price change (40%)
    price_change = (current_close - close_96_ago) / close_96_ago
    price_score = min(1.0, max(0.0, (price_change + 0.20) / 0.40))  # -20%→0, +20%→1

    # 2. 24h volume surge (20%)
    recent_vols = volumes[-96:] if len(volumes) >= 96 else volumes
    avg_vol = sum(recent_vols) / len(recent_vols) if recent_vols else 1
    vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
    vol_score = min(1.0, vol_ratio / 3.0)  # 3x average → 1.0

    # 3. RSI-14 normalized (20%)
    rsi_val = rsi(closes, 14)
    # RSI 30-70 is healthy → score 0.5-0.8. Below 30 = oversold (0.2), above 70 = overbought (0.6)
    if rsi_val <= 30:
        rsi_score = 0.2 + (rsi_val / 30) * 0.3
    elif rsi_val <= 70:
        rsi_score = 0.5 + ((rsi_val - 30) / 40) * 0.3
    else:
        rsi_score = 0.8 - ((rsi_val - 70) / 30) * 0.2
    rsi_score = max(0.0, min(1.0, rsi_score))

    # 4. EMA50 ratio (20%)
    ema50 = ema(closes[-50:], 50) if len(closes) >= 50 else current_close
    ema_ratio = (current_close - ema50) / ema50 if ema50 > 0 else 0.0
    ema_score = min(1.0, max(0.0, (ema_ratio + 0.10) / 0.20))  # -10%→0, +10%→1

    composite = (
        0.40 * price_score +
        0.20 * vol_score +
        0.20 * rsi_score +
        0.20 * ema_score
    )

    return MomentumCandidate(
        symbol=symbol,
        composite_score=round(composite, 4),
        price_change_24h_pct=round(price_change * 100, 2),
        volume_change_24h_pct=round((vol_ratio - 1) * 100, 2),
        rsi_4h=round(rsi_val, 1),
        close_vs_ema50_pct=round(ema_ratio * 100, 2),
    )


def momentum_discover(
    klines: dict[str, list[dict]],
    tick_index: int,
    regime: str,
    top_n: int = 5,
) -> list[MomentumCandidate]:
    """Discover top momentum candidates at a given tick from k-line data."""
    candidates: list[MomentumCandidate] = []
    for sym, candles in klines.items():
        if not is_eligible(sym) or is_stablecoin(sym) or sym == "BNB":
            continue
        cand = compute_momentum(sym, candles, tick_index)
        if cand is not None and cand.composite_score > 0.35:
            candidates.append(cand)

    candidates.sort(key=lambda c: c.composite_score, reverse=True)

    # In risk_off, tighter filter
    if regime == "risk_off":
        candidates = [c for c in candidates if c.composite_score > 0.55]

    return candidates[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
# Regime Classification (rule-based — no LLM)
# ═══════════════════════════════════════════════════════════════════════════

def classify_regime_backtest(fear_greed: dict[str, int], date_str: str, tick_hour: int) -> tuple[str, dict]:
    """
    Regime from Fear & Greed daily value using the same rules as the LLM prompt.

    Returns (regime, params_dict).
    Falls back to "neutral/3pos/0.35" when F&G data is missing.
    """
    fg = fear_greed.get(date_str, 50)

    if fg > 75:
        regime = "risk_off"
        params = {"max_positions": 1, "allocation_pct": 0.15, "momentum_lookback": "24h"}
    elif fg < 20:
        regime = "risk_off"
        params = {"max_positions": 1, "allocation_pct": 0.15, "momentum_lookback": "24h"}
    elif 35 <= fg <= 65:
        regime = "risk_on"
        params = {"max_positions": 5, "allocation_pct": 0.50, "momentum_lookback": "4h"}
    else:
        regime = "neutral"
        params = {"max_positions": 3, "allocation_pct": 0.35, "momentum_lookback": "24h"}

    return regime, params


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio Simulation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SimState:
    """Mutable simulation state for one backtest run."""
    usdt_balance: float = INITIAL_USD
    holdings: dict[str, float] = field(default_factory=dict)  # {SYMBOL: token_amount}
    peak_value: float = INITIAL_USD
    trades_today: int = 0
    trade_date: str = ""

    def total_value(self, prices: dict[str, float]) -> float:
        total = self.usdt_balance
        for sym, amt in self.holdings.items():
            total += amt * prices.get(sym.upper(), 0)
        return total

    def holdings_dict(self, prices: dict[str, float]) -> dict:
        """Build the {symbol: {balance, cost_basis_usd}} dict that portfolio.py expects."""
        h: dict[str, dict] = {}
        for sym, amt in self.holdings.items():
            px = prices.get(sym.upper(), 0)
            h[sym] = {"balance": amt, "cost_basis_usd": amt * px}
        # Add USDT as a "holding" for cost basis tracking
        h["USDT"] = {"balance": self.usdt_balance, "cost_basis_usd": self.usdt_balance}
        # Simulate a small BNB balance for gas buffer (never sold)
        if "BNB" not in h:
            h["BNB"] = {"balance": 0.05, "cost_basis_usd": 25.0}
        return h


def execute_swap_sim(swap: SwapInstruction, state: SimState, prices: dict[str, float]) -> bool:
    """
    Simulate a swap at current prices with slippage haircut.

    SELL: remove from_token balance, add to_token (USDT) * slippage
    BUY:  remove USDT, add to_token amount * slippage

    Returns True on success.
    """
    from_tok = swap.from_token.upper()
    to_tok = swap.to_token.upper()
    from_px = prices.get(from_tok, 1.0)
    to_px = prices.get(to_tok, 1.0)

    if swap.action == "sell":
        # Sell exact token balance (as guaranteed by portfolio.py)
        amt = swap.amount_token
        if from_tok not in state.holdings or state.holdings[from_tok] < amt:
            return False
        state.holdings[from_tok] -= amt
        if state.holdings[from_tok] < 1e-12:
            del state.holdings[from_tok]
        # Receive USDT (or to_token) with slippage
        received = amt * from_px * SLIPPAGE_MULTIPLIER
        if to_tok == "USDT":
            state.usdt_balance += received
        else:
            state.holdings[to_tok] = state.holdings.get(to_tok, 0) + received / to_px

    elif swap.action == "buy":
        # Pay with USDT
        cost_usd = swap.amount_usd
        if state.usdt_balance < cost_usd:
            return False
        state.usdt_balance -= cost_usd
        received = (cost_usd / to_px) * SLIPPAGE_MULTIPLIER
        state.holdings[to_tok] = state.holdings.get(to_tok, 0) + received

    return True


def _prices_at_tick(klines: dict[str, list[dict]], tick: int) -> dict[str, float]:
    """Extract closing prices from k-line data at a given tick index."""
    px: dict[str, float] = {"USDT": 1.0, "FDUSD": 1.0}
    for sym, candles in klines.items():
        if tick < len(candles):
            close = candles[tick].get("quote", {}).get("USD", {}).get("close", 0)
            if close > 0:
                px[sym.upper()] = close
    return px


# ═══════════════════════════════════════════════════════════════════════════
# Main Backtest Loop
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestReport:
    start_value: float
    end_value: float
    total_pnl_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    total_trades: int
    trades_per_day: float
    win_rate_pct: float
    avg_return_per_trade_pct: float
    peak_value: float
    trough_value: float
    eq_curve: list[float] = field(default_factory=list)

    def print(self):
        print()
        print("=" * 60)
        print("  BACKTEST REPORT")
        print("=" * 60)
        print(f"  Start value:        ${self.start_value:,.0f}")
        print(f"  End value:          ${self.end_value:,.0f}")
        print(f"  ─────────────────────────────────")
        print(f"  Total PnL:          {self.total_pnl_pct:+.1f}%")
        print(f"  Max drawdown:       {self.max_drawdown_pct:.1f}%")
        print(f"  Sharpe ratio:       {self.sharpe_ratio:.2f}")
        print(f"  Peak value:         ${self.peak_value:,.0f}")
        print(f"  Trough value:       ${self.trough_value:,.0f}")
        print(f"  ─────────────────────────────────")
        print(f"  Total trades:       {self.total_trades}")
        print(f"  Trades/day:         {self.trades_per_day:.1f}")
        print(f"  Win rate:           {self.win_rate_pct:.1f}%")
        print(f"  Avg return/trade:   {self.avg_return_per_trade_pct:+.2f}%")
        print("=" * 60)


def run_backtest(
    klines: dict[str, list[dict]],
    fear_greed: dict[str, int],
    interval: str = DEFAULT_INTERVAL,
) -> BacktestReport:
    """
    Run the strategy pipeline across the k-line time series.

    Each tick:
    1. Extract prices at this timestamp
    2. Determine regime from F&G
    3. Discover momentum candidates from k-line data
    4. Generate swap plan via portfolio.generate_swap_plan()
    5. Simulate execution, track state
    """
    state = SimState()
    eq_curve: list[float] = [INITIAL_USD]

    # Find common time range
    max_ticks = max((len(c) for c in klines.values()), default=0)

    if max_ticks < 96:
        log.error("Not enough k-line data (need ≥96 periods for 24h lookback)")
        return BacktestReport(INITIAL_USD, INITIAL_USD, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    total_trades = 0
    winning_trades = 0
    trade_returns: list[float] = []
    daily_returns: list[float] = []
    peak_value = INITIAL_USD
    trough_value = INITIAL_USD
    max_drawdown_pct = 0.0
    prev_day_value = INITIAL_USD
    last_date = ""

    # We tick at the interval cadence — each candle is one tick
    # Skip the first 96 candles (warm-up for 24h lookback)
    for tick in range(96, max_ticks):
        prices = _prices_at_tick(klines, tick)
        current_value = state.total_value(prices)

        # Track peak/trough/drawdown
        if current_value > peak_value:
            peak_value = current_value
        if current_value < trough_value:
            trough_value = current_value
        dd = (peak_value - current_value) / peak_value if peak_value > 0 else 0
        if dd > max_drawdown_pct:
            max_drawdown_pct = dd

        eq_curve.append(current_value)

        # ── Get date for this tick ──
        # Use the first token's timestamp
        first_candles = next(iter(klines.values()), [])
        tick_ts = first_candles[tick].get("timestamp", "") if tick < len(first_candles) else ""
        date_str = tick_ts[:10] if tick_ts else ""
        try:
            hour = int(tick_ts[11:13]) if len(tick_ts) >= 13 else 0
        except (ValueError, IndexError):
            hour = 0

        # ── Daily trade counter reset ──
        if date_str != state.trade_date:
            state.trades_today = 0
            state.trade_date = date_str
            # Track daily PnL
            if prev_day_value > 0 and last_date:
                daily_returns.append((current_value - prev_day_value) / prev_day_value)
            prev_day_value = current_value
            last_date = date_str

        if state.trades_today >= MAX_TRADES_PER_DAY:
            continue  # quota exhausted

        # ── 25% drawdown emergency: sell all volatile → USDT ──
        if dd >= 0.25:
            # Emergency sell: liquidate all non-stable, non-BNB holdings
            for sym in list(state.holdings.keys()):
                if sym == "BNB" or is_stablecoin(sym):
                    continue
                amt = state.holdings.pop(sym, 0)
                px = prices.get(sym.upper(), 0)
                state.usdt_balance += amt * px * SLIPPAGE_MULTIPLIER
            continue

        # ── Regime ──
        regime, params = classify_regime_backtest(fear_greed, date_str, hour)

        # ── Momentum discovery ──
        candidates = momentum_discover(klines, tick, regime, top_n=5)
        if not candidates:
            continue

        # ── Portfolio: generate swap plan ──
        portfolio_total = state.total_value(prices)
        holdings_for_plan = state.holdings_dict(prices)

        price_cache = {sym.upper(): px for sym, px in prices.items()}
        price_cache.setdefault("USDT", 1.0)
        price_cache.setdefault("FDUSD", 1.0)

        plan: SwapPlan = generate_swap_plan(
            holdings=holdings_for_plan,
            candidates=candidates,
            price_cache=price_cache,
            regime=regime,
            max_positions=params["max_positions"],
            allocation_pct=params["allocation_pct"],
            total_value_usd=portfolio_total,
            trades_today=state.trades_today,
        )

        if not plan.swaps:
            continue

        # ── Execute swaps ──
        for swap in plan.swaps:
            success = execute_swap_sim(swap, state, prices)
            if success:
                state.trades_today += 1
                total_trades += 1

                # Track trade PnL: compare entry price to a simple metric
                # A "win" = bought token outperforms USDT over the next 24h
                if swap.action == "buy":
                    to_tok = swap.to_token.upper()
                    future_tick = min(tick + 96, max_ticks - 1)
                    future_prices = _prices_at_tick(klines, future_tick)
                    entry_px = prices.get(to_tok, 1.0)
                    exit_px = future_prices.get(to_tok, entry_px)
                    ret = (exit_px - entry_px) / entry_px if entry_px > 0 else 0
                    trade_returns.append(ret)
                    if ret > 0:
                        winning_trades += 1

    # ── Final valuation ──
    final_prices = _prices_at_tick(klines, max_ticks - 1)
    final_value = state.total_value(final_prices)
    if last_date and final_value > 0:
        daily_returns.append((final_value - prev_day_value) / prev_day_value)

    eq_curve.append(final_value)

    total_pnl_pct = (final_value - INITIAL_USD) / INITIAL_USD * 100

    # Sharpe ratio (annualized, assuming ~365 crypto trading days)
    if daily_returns:
        avg_daily = sum(daily_returns) / len(daily_returns)
        variance = sum((r - avg_daily) ** 2 for r in daily_returns) / len(daily_returns)
        std_daily = math.sqrt(variance) if variance > 0 else 0.0001
        sharpe = (avg_daily / std_daily) * math.sqrt(365) if std_daily > 0 else 0
    else:
        sharpe = 0.0
        avg_daily = 0.0

    trades_per_day = total_trades / max(1, len(daily_returns))
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    avg_ret = (sum(trade_returns) / len(trade_returns) * 100) if trade_returns else 0.0

    return BacktestReport(
        start_value=INITIAL_USD,
        end_value=final_value,
        total_pnl_pct=round(total_pnl_pct, 2),
        max_drawdown_pct=round(max_drawdown_pct * 100, 2),
        sharpe_ratio=round(sharpe, 3),
        total_trades=total_trades,
        trades_per_day=round(trades_per_day, 2),
        win_rate_pct=round(win_rate, 1),
        avg_return_per_trade_pct=round(avg_ret, 2),
        peak_value=peak_value,
        trough_value=trough_value,
        eq_curve=eq_curve,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Backtest — replay 30 days through the momentum rotation strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/backtest.py                        default: 30d, 30 tokens, 15m
  python3 scripts/backtest.py --days 7 --top 10     quick sanity check
  python3 scripts/backtest.py --interval 1h         hourly instead of 15-min
  python3 scripts/backtest.py --all-tokens           full 149-token allowlist
  python3 scripts/backtest.py --tokens BTC,ETH,BNB   specific tokens only
  python3 scripts/backtest.py --cache-only           just download data, skip sim
        """,
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"Days of history to replay (default: {DEFAULT_DAYS})")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL,
                        choices=["5m", "15m", "30m", "1h", "4h", "1d"],
                        help=f"Candle interval (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N,
                        help=f"Number of tokens to include (default: {DEFAULT_TOP_N})")
    parser.add_argument("--all-tokens", action="store_true",
                        help="Include all 149 allowlist tokens")
    parser.add_argument("--tokens",
                        help="Comma-separated token list (e.g., BTC,ETH,CAKE)")
    parser.add_argument("--cache-only", action="store_true",
                        help="Download k-line data to cache, then exit")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cache, re-download everything")
    parser.add_argument("--no-llm", action="store_true", default=True,
                        help="Use rule-based regime (no LLM calls) — default for backtest")
    args = parser.parse_args()

    # ── Token selection ──
    if args.tokens:
        symbols = [s.strip().upper() for s in args.tokens.split(",")]
    elif args.all_tokens:
        symbols = list(COMP_TOKENS)
    else:
        # Default: top N liquid tokens from allowlist
        # Prioritize known liquid tokens, then fill from allowlist
        liquid = ["BTC", "ETH", "BNB", "CAKE", "LINK", "DOGE", "ADA", "XRP", "DOT",
                   "UNI", "AVAX", "MATIC", "SOL", "FIL", "ATOM", "NEAR", "OP",
                   "ARB", "PEPE", "SHIB", "FLOKI", "INJ", "RUNE", "APT", "SUI",
                   "GMX", "LDO", "STX", "ENS", "AAVE"]
        symbols = [s for s in liquid if s in COMP_TOKENS][:args.top]
        # Pad from allowlist if needed
        for t in COMP_TOKENS:
            if len(symbols) >= args.top:
                break
            if t not in symbols:
                symbols.append(t)

    if not os.getenv("CMC_API_KEY"):
        print("❌ CMC_API_KEY env var is required for backtest data download.")
        print("   export CMC_API_KEY=your-key")
        sys.exit(1)

    # ── Clear cache if requested ──
    if args.no_cache:
        cache_path = CACHE_DIR / f"klines_{args.interval}_{args.days}d.json"
        fg_path = CACHE_DIR / "fear_greed.json"
        for p in [cache_path, fg_path]:
            if p.exists():
                p.unlink()
        log.info("Cache cleared")

    # ── Download data ──
    log.info("Downloading %d tokens × %dd of %s k-line data...",
             len(symbols), args.days, args.interval)
    klines = download_klines(symbols, interval=args.interval, days=args.days)
    fear_greed = download_fear_greed(args.days)

    if args.cache_only:
        print(f"✅ K-line data cached for {len(klines)} tokens at {CACHE_DIR}")
        return

    if len(klines) < 3:
        print(f"❌ Only {len(klines)} tokens downloaded — need at least 3 for backtest.")
        print("   Check CMC_API_KEY validity or try fewer tokens with --tokens.")
        sys.exit(1)

    # ── Run backtest ──
    log.info("Running backtest on %d tokens, %d days of %s candles...",
             len(klines), args.days, args.interval)

    report = run_backtest(klines, fear_greed, interval=args.interval)
    report.print()

    # Buy-and-hold benchmark
    bnbs = [c.get("quote", {}).get("USD", {}).get("close", 0)
            for c in klines.get("BNB", [])]
    if bnbs and bnbs[-1] > 0 and bnbs[96] > 0:
        bnb_pnl = (bnbs[-1] - bnbs[96]) / bnbs[96] * 100
        print(f"\n  BNB buy-and-hold:   {bnb_pnl:+.1f}%  (benchmark)")
        print(f"  Alpha vs BNB:       {report.total_pnl_pct - bnb_pnl:+.1f}%")

    btc_candles = klines.get("BTC", [])
    if btc_candles:
        btc_closes = [c.get("quote", {}).get("USD", {}).get("close", 0) for c in btc_candles]
        if len(btc_closes) > 96 and btc_closes[-1] > 0 and btc_closes[96] > 0:
            btc_pnl = (btc_closes[-1] - btc_closes[96]) / btc_closes[96] * 100
            print(f"  BTC buy-and-hold:   {btc_pnl:+.1f}%  (benchmark)")

    print()


if __name__ == "__main__":
    main()
