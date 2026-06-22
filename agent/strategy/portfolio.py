"""
Portfolio manager — position sizing, trailing stops, and swap plan generation.

Called after momentum discovery. Compares current holdings against top
momentum candidates, applies trailing stop-loss, and generates a sell-first
swap plan truncated to the daily trade quota.

Exit strategy (trailing stop):
- Tracks peak price per token since entry.
- If price drops 5% below the peak → full exit + 2h penalty box.
- No fixed take-profit — let winners ride until they reverse.

Position sizing:
- Concentrated: top 2 targets, 50% allocation each.
- Truncated to daily quota (5 trades/day).

Circuit breaker note:
- At -25% from portfolio peak, main.py halts all entries.
- Only stop-loss exits and heartbeat trades are allowed.
"""

import logging
import os
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from data.allowlist import is_stablecoin

log = logging.getLogger(__name__)

# ── Competition limits ──
MAX_TRADES_PER_DAY = int(os.getenv("AGENT_MAX_TRADES", "16"))

# ── Safety constants ──
SLIPPAGE_MULTIPLIER = 0.985   # 1.5% haircut: 0.25% DEX fee + slippage buffer
BNB_GAS_BUFFER_USD = 20.0     # minimum BNB to leave for gas (never sell below this)

# ── Trailing stop ──
TRAILING_STOP_PCT = 0.05      # 5% below peak price → full exit

# ── Penalty box ──
COOLDOWN_SECONDS = 7200       # 2 hours — prevents re-buying a stopped-out token

# ── Minimum hold duration (anti-churn) ──
MIN_HOLD_SECONDS = 1800       # 30 min — don't sell a position for rotation if held < this


# ── Data models ──

@dataclass
class SwapInstruction:
    action: str           # "buy" or "sell"
    from_token: str       # symbol (exact case)
    to_token: str         # symbol (exact case)
    amount_usd: float     # estimated USD value
    amount_token: float = 0.0  # exact token units (sell=wallet balance, buy=USD/price)
    reason: str = ""


@dataclass
class SwapPlan:
    swaps: list[SwapInstruction] = field(default_factory=list)
    trades_used: int = 0
    remaining_quota: int = MAX_TRADES_PER_DAY
    idle_capital_usd: float = 0.0
    note: str = ""
    new_cooldowns: dict[str, float] = field(default_factory=dict)
    # {symbol: cooldown_until_ts} — tokens stopped out this tick


# ── Main entry point ──

def generate_swap_plan(
    holdings: dict,            # {symbol: {balance, cost_basis_usd, peak_price?, pool_address?}}
    candidates: list,          # list[MomentumCandidate] from momentum.py
    price_cache: dict,         # {symbol: float} — USD price per token from CMC
    regime: str,               # "risk_on" | "neutral" | "risk_off"
    max_positions: int,        # from regime decision (default 2 for concentrated)
    allocation_pct: float,     # from regime decision
    total_value_usd: float,    # total portfolio value in USD
    trades_today: int,         # trades already executed today
) -> SwapPlan:
    """
    Generate a rebalancing swap plan with trailing stops.

    Algorithm:
    1. Trailing stop scan: check each holding against its peak price.
       If price ≤ peak × 0.95 → full exit + penalty box cooldown.
    2. Build target set from top-N candidates.
    3. Sell holdings not in target set (skip already stopped-out).
    4. Buy target tokens at equal weight.
    5. Order: exits first, then regular sells, then buys.
    """
    remaining = MAX_TRADES_PER_DAY - trades_today
    if remaining <= 0:
        log.info("Daily trade quota exhausted (%d/%d)", trades_today, MAX_TRADES_PER_DAY)
        return SwapPlan(remaining_quota=0, note="Daily trade quota exhausted")

    now = _time.time()
    plan = SwapPlan(remaining_quota=remaining)

    # ── Classify holdings ──
    stable_balance_usd = 0.0
    volatile_holdings: dict[str, dict] = {}
    bnb_holdings_usd = 0.0
    bnb_balance = 0.0

    for sym, info in holdings.items():
        if is_stablecoin(sym):
            stable_balance_usd += info.get("cost_basis_usd", 0)
        elif sym == "BNB":
            bnb_balance = info.get("balance", 0.0)
            bnb_holdings_usd = info.get("cost_basis_usd", 0.0)
        else:
            volatile_holdings[sym] = info

    # ════════════════════════════════════════════════════════════════
    # EXIT PRE-SCAN: Trailing Stop-Loss
    # "Buy high, sell higher" — no take-profit, ride the pump.
    # Exit only when price drops 5% below the peak seen since entry.
    # On exit → penalty box cooldown for 2 hours.
    # ════════════════════════════════════════════════════════════════
    exit_swaps: list[SwapInstruction] = []

    for sym, info in volatile_holdings.items():
        balance = info.get("balance", 0.0)
        if balance <= 0:
            continue

        current_price = price_cache.get(sym, 0)
        if current_price <= 0:
            continue

        # Use per-token peak_price if stored, otherwise cost_basis as initial peak
        stored_peak = info.get("peak_price", 0)
        cost_basis = info.get("cost_basis_usd", current_price)
        if stored_peak > 0:
            peak = stored_peak
        elif cost_basis > 0:
            peak = cost_basis
        else:
            peak = current_price  # fresh position, no history

        # Update peak if we've gone higher
        if current_price > peak:
            peak = current_price

        # Trailing stop: exit when current < peak × (1 - TRAILING_STOP_PCT)
        stop_price = peak * (1.0 - TRAILING_STOP_PCT)
        if current_price <= stop_price:
            amount_usd = balance * current_price
            exit_swaps.append(SwapInstruction(
                action="sell",
                from_token=sym,
                to_token="USDT",
                amount_usd=amount_usd,
                amount_token=balance,
                reason=(
                    f"TRAILING_STOP: peak=${peak:.4f} "
                    f"now=${current_price:.4f} ({((current_price/peak)-1)*100:+.1f}%) "
                    f"cooldown={COOLDOWN_SECONDS}s"
                ),
            ))
            plan.new_cooldowns[sym] = now + COOLDOWN_SECONDS
            log.warning(
                "Trailing stop: %s peak=$%.4f → now=$%.4f (%.1f%% below peak). "
                "Penalty box until %s.",
                sym, peak, current_price, (1 - current_price / peak) * 100,
                _time.strftime("%H:%M:%S", _time.localtime(now + COOLDOWN_SECONDS)),
            )

    # Exit swaps consume quota
    sq = exit_swaps[:remaining]
    exit_quota_used = len(sq)

    # Track tokens already fully closed (skip regular sells)
    exited_tokens: set[str] = {s.from_token for s in sq}

    # ════════════════════════════════════════════════════════════════
    # Build target set
    # ════════════════════════════════════════════════════════════════
    target_symbols = [c.symbol for c in candidates[:max_positions]]
    target_set = set(target_symbols)

    # Position sizing: equal weight within allocation
    deployable = total_value_usd * allocation_pct
    per_position = deployable / max(len(target_set), 1)

    # ════════════════════════════════════════════════════════════════
    # REGULAR SELL candidates: holdings NOT in target set
    # ════════════════════════════════════════════════════════════════
    sell_candidates: list[SwapInstruction] = []
    for sym, info in volatile_holdings.items():
        if sym in exited_tokens:
            continue
        if sym not in target_set:
            # Skip recently-bought positions (anti-churn: give the trade time to develop)
            entry_ts_str = info.get("entry_ts", "")
            if entry_ts_str:
                try:
                    if entry_ts_str.endswith("Z"):
                        entry_dt = datetime.fromisoformat(entry_ts_str.replace("Z", "+00:00"))
                    else:
                        entry_dt = datetime.fromisoformat(entry_ts_str)
                    age_sec = now - entry_dt.timestamp()
                    if age_sec < MIN_HOLD_SECONDS:
                        log.info("  Holding %s: %ds old (< %ds min hold) — skipping rotation sell",
                                 sym, int(age_sec), MIN_HOLD_SECONDS)
                        continue
                except (ValueError, TypeError):
                    pass  # can't parse entry_ts, proceed with sell
            cost_usd = info.get("cost_basis_usd", 0)
            balance = info.get("balance", 0.0)
            if cost_usd <= 0 or balance <= 0:
                continue
            sell_candidates.append(SwapInstruction(
                action="sell",
                from_token=sym,
                to_token="USDT",
                amount_usd=cost_usd,
                amount_token=balance,
                reason=f"Not in target set (targets={target_set})",
            ))

    # BNB gas buffer
    if "BNB" not in target_set and bnb_holdings_usd > BNB_GAS_BUFFER_USD:
        excess_usd = bnb_holdings_usd - BNB_GAS_BUFFER_USD
        if excess_usd > 5.0 and bnb_balance > 0:
            bnb_price = price_cache.get("BNB", 0)
            if bnb_price > 0:
                excess_balance = excess_usd / bnb_price
                if excess_balance > 0:
                    sell_candidates.append(SwapInstruction(
                        action="sell",
                        from_token="BNB",
                        to_token="USDT",
                        amount_usd=excess_usd,
                        amount_token=excess_balance,
                        reason=f"BNB excess above ${BNB_GAS_BUFFER_USD} gas buffer",
                    ))

    remaining_after_exits = remaining - exit_quota_used

    sell_candidates.sort(key=lambda s: s.amount_usd, reverse=True)
    surviving_sells = sell_candidates[:remaining_after_exits]

    raw_freed = sum(s.amount_usd for s in surviving_sells)
    freed_capital = raw_freed * SLIPPAGE_MULTIPLIER
    remaining_after_sells = remaining_after_exits - len(surviving_sells)

    exit_freed = sum(s.amount_usd for s in sq)
    available_usd = stable_balance_usd + freed_capital + (exit_freed * SLIPPAGE_MULTIPLIER)

    # ════════════════════════════════════════════════════════════════
    # BUY candidates: concentrated equal weight
    # ════════════════════════════════════════════════════════════════
    buy_candidates: list[SwapInstruction] = []
    for sym in target_symbols:
        if sym == "BNB":
            continue
        current_value = volatile_holdings.get(sym, {}).get("cost_basis_usd", 0)
        needed = per_position - current_value
        if needed <= 0:
            continue

        buy_amount_usd = min(needed, available_usd)
        if buy_amount_usd < 5.0:
            continue

        price = price_cache.get(sym, 0)
        if price <= 0:
            log.warning("No price for %s — skipping buy", sym)
            continue
        buy_amount_token = buy_amount_usd / price

        buy_candidates.append(SwapInstruction(
            action="buy",
            from_token="USDT",
            to_token=sym,
            amount_usd=buy_amount_usd,
            amount_token=buy_amount_token,
            reason=f"Momentum target, score={_find_score(candidates, sym):.3f}",
        ))

    surviving_buys = buy_candidates[:remaining_after_sells]

    # ── Assemble plan: exits → sells → buys ──
    plan.swaps = sq + surviving_sells + surviving_buys
    plan.trades_used = len(plan.swaps)
    plan.remaining_quota = remaining - plan.trades_used
    spent_on_buys_usd = sum(s.amount_usd for s in surviving_buys)
    plan.idle_capital_usd = available_usd - spent_on_buys_usd

    total_candidates = len(sell_candidates + buy_candidates)
    total_surviving = len(plan.swaps)
    if total_candidates > total_surviving:
        dropped_sells = sell_candidates[len(surviving_sells):]
        dropped_buys = buy_candidates[len(surviving_buys):]
        dropped = dropped_sells + dropped_buys
        plan.note = (
            f"Quota: {trades_today}+{total_surviving}={trades_today+total_surviving}/{MAX_TRADES_PER_DAY}. "
            f"Dropped {len(dropped)}: {[(s.action, s.to_token) for s in dropped]}."
        )
    if raw_freed > 0:
        plan.note += f" Slippage: ${raw_freed:.0f}→${freed_capital:.0f} (×{SLIPPAGE_MULTIPLIER})."

    log.info(
        "Swap plan: %d exits + %d sells + %d buys → %d executed (%d quota remaining). "
        "Idle=%.0f USD. %s",
        len(sq), len(surviving_sells), len(surviving_buys), plan.trades_used,
        plan.remaining_quota, plan.idle_capital_usd, plan.note or "ok",
    )
    return plan


def _find_score(candidates: list, symbol: str) -> float:
    """Find the composite/acceleration score for a symbol in the candidate list."""
    for c in candidates:
        if c.symbol == symbol:
            return getattr(c, "composite_score", 0) or getattr(c, "acceleration_score", 0)
    return 0.0
