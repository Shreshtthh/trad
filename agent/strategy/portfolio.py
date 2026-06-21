"""
Portfolio manager — position sizing, rebalancing, and swap plan generation.

Called after momentum discovery. Compares current holdings against top
momentum candidates, generates a sell-first swap plan, and truncates to
the daily trade quota.

Trade counting: Each token-A → token-B swap counts as ONE rebalance.
PancakeSwap may route through intermediate pairs internally (e.g.,
OBSCURE → WBNB → USDT → TARGET), but TWAK's swap command abstracts this
into one intentional trade. We count rebalances, not router hops.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from data.allowlist import is_stablecoin

log = logging.getLogger(__name__)

# ── Competition limits ──
MAX_TRADES_PER_DAY = 5

# ── Safety constants ──
SLIPPAGE_MULTIPLIER = 0.985   # 1.5% haircut: 0.25% DEX fee + slippage buffer
BNB_GAS_BUFFER_USD = 20.0     # minimum BNB to leave for gas (never sell below this)

# ── Exit guardrails ──
STOP_LOSS_PCT = -0.05         # -5% from cost basis → force-sell 100% of position
TAKE_PROFIT_PCT = 0.08        # +8% from cost basis → sell 50% to lock in gains
# Round-trip friction is ~3% (1.5% entry + 1.5% exit).
# Stop at -5% limits worst-case to -8% net loss.
# Take-profit at +8% locks in ~5% net gain after friction.


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


# ── Main entry point ──

def generate_swap_plan(
    holdings: dict,            # {symbol(str): {"balance": float, "cost_basis_usd": float}}
    candidates: list,          # list[MomentumCandidate] from momentum.py
    price_cache: dict,         # {symbol(str): float} — USD price per token from CMC
    regime: str,               # "risk_on" | "neutral" | "risk_off"
    max_positions: int,        # from regime decision
    allocation_pct: float,     # from regime decision (fraction of portfolio to deploy)
    total_value_usd: float,    # total portfolio value in USD
    trades_today: int,         # trades already executed today
) -> SwapPlan:
    """
    Generate a rebalancing swap plan with exact token amounts.

    Algorithm:
    1. Compute target allocations: top-N candidates, equal weight within allocation_pct.
    2. Reserve BNB gas buffer ($20) — never sell BNB below this threshold.
    3. Identify EXITs: holdings not in target set → sell to USDT (exact wallet balance).
    4. Apply slippage multiplier (0.985) to freed capital.
    5. Identify ENTRYs: target tokens below target weight → buy with exact token amounts.
    6. Order: all SELLs first, then BUYs. Truncate to daily quota.
    """
    remaining = MAX_TRADES_PER_DAY - trades_today
    if remaining <= 0:
        log.info("Daily trade quota exhausted (%d/%d)", trades_today, MAX_TRADES_PER_DAY)
        return SwapPlan(remaining_quota=0, note="Daily trade quota exhausted")

    plan = SwapPlan(remaining_quota=remaining)

    # ── Classify holdings: stablecoins, volatile, BNB gas reserve ──
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

    # ── Build target set ──
    target_symbols = [c.symbol for c in candidates[:max_positions]]
    target_set = set(target_symbols)

    # ── Position size per target ──
    deployable = total_value_usd * allocation_pct
    per_position = deployable / max(len(target_set), 1)

    # ════════════════════════════════════════════════════════════════
    # EXIT PRE-SCAN: stop-loss and take-profit (run BEFORE regular sells).
    # These get priority — a stop-loss on a bleeding position matters more
    # than rebalancing into a new target.
    # ════════════════════════════════════════════════════════════════
    exit_swaps: list[SwapInstruction] = []

    for sym, info in volatile_holdings.items():
        cost_basis = info.get("cost_basis_usd", 0)
        balance = info.get("balance", 0.0)
        if cost_basis <= 0 or balance <= 0:
            continue

        current_price = price_cache.get(sym, 0)
        if current_price <= 0:
            continue

        # Unrealized PnL from cost basis
        pnl_pct = (current_price - cost_basis) / cost_basis

        if pnl_pct <= STOP_LOSS_PCT:
            # Full exit — cut the loss at -5%
            exit_swaps.append(SwapInstruction(
                action="sell",
                from_token=sym,
                to_token="USDT",
                amount_usd=cost_basis,
                amount_token=balance,
                reason=f"STOP_LOSS: {pnl_pct:+.1%} from cost basis (limit={STOP_LOSS_PCT:+.0%})",
            ))
            log.warning("Stop-loss triggered: %s at %+.1f%% (cost=$%.2f, now=$%.2f)",
                       sym, pnl_pct * 100, cost_basis, current_price)

        elif pnl_pct >= TAKE_PROFIT_PCT:
            # Half exit — lock in gains, let remainder ride
            half_balance = balance * 0.5
            half_cost = cost_basis * 0.5
            exit_swaps.append(SwapInstruction(
                action="sell",
                from_token=sym,
                to_token="USDT",
                amount_usd=half_cost,
                amount_token=half_balance,
                reason=f"TAKE_PROFIT: {pnl_pct:+.1%} from cost basis (target={TAKE_PROFIT_PCT:+.0%}, selling 50%)",
            ))
            log.info("Take-profit triggered: %s at %+.1f%% — selling half",
                    sym, pnl_pct * 100)

    # Exit swaps consume quota
    sq = exit_swaps[:remaining]
    exit_quota_used = len(sq)

    # Track tokens fully closed by exit logic (so regular sells don't duplicate)
    # Stop-loss: entire position is sold → skip regular sell entirely.
    # Take-profit: half is sold → skip regular sell for remainder (let it ride).
    exited_tokens: set[str] = set()
    for s in sq:
        exited_tokens.add(s.from_token)

    # ════════════════════════════════════════════════════════════════
    # REGULAR SELL candidates: holdings NOT in target set
    # Skip tokens already closed by stop-loss.
    # ════════════════════════════════════════════════════════════════
    sell_candidates: list[SwapInstruction] = []
    for sym, info in volatile_holdings.items():
        if sym in exited_tokens:
            continue  # Already handled by stop-loss or take-profit
        if sym not in target_set:
            cost_usd = info.get("cost_basis_usd", 0)
            balance = info.get("balance", 0.0)
            if cost_usd <= 0 or balance <= 0:
                continue

            # Convert to exact token amount: sell FULL wallet balance
            # (no dust left behind — TWAK executes on exact token units)
            sell_candidates.append(SwapInstruction(
                action="sell",
                from_token=sym,
                to_token="USDT",
                amount_usd=cost_usd,
                amount_token=balance,
                reason=f"Not in target set (targets={target_set})",
            ))

    # BNB gas buffer: only sell BNB EXCESS above $20, and only if not in target set
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

    # Quota already consumed by exit swaps (stop-loss/take-profit run first)
    remaining_after_exits = remaining - exit_quota_used

    # Regular sells consume remaining quota — sort by value descending, take top-N
    sell_candidates.sort(key=lambda s: s.amount_usd, reverse=True)
    surviving_sells = sell_candidates[:remaining_after_exits]

    # Freed capital from sells, with slippage haircut
    raw_freed = sum(s.amount_usd for s in surviving_sells)
    freed_capital = raw_freed * SLIPPAGE_MULTIPLIER
    remaining_after_sells = remaining_after_exits - len(surviving_sells)

    # Available capital: stablecoins + slippage-adjusted sell proceeds
    # Note: stop-loss/take-profit capital is also freed and added here
    exit_freed = sum(s.amount_usd for s in sq)
    available_usd = stable_balance_usd + freed_capital + (exit_freed * SLIPPAGE_MULTIPLIER)

    # ── BUY candidates: target tokens at equal weight ──
    buy_candidates: list[SwapInstruction] = []
    for sym in target_symbols:
        current_value = volatile_holdings.get(sym, {}).get("cost_basis_usd", 0)
        if sym == "BNB":
            continue  # BNB handled separately via gas buffer, not a momentum target

        needed = per_position - current_value
        if needed <= 0:
            continue

        buy_amount_usd = min(needed, available_usd)
        if buy_amount_usd < 5.0:
            continue

        # Convert USD to exact token units
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
            reason=f"Momentum target, composite={_find_score(candidates, sym):.3f}",
        ))

    # Buys consume quota — truncate to remaining_after_sells
    surviving_buys = buy_candidates[:remaining_after_sells]

    # ── Build final plan: exits first, then regular sells, then buys ──
    plan.swaps = sq + surviving_sells + surviving_buys
    plan.trades_used = len(plan.swaps)
    plan.remaining_quota = remaining - plan.trades_used

    # idle_capital_usd: available capital minus SURVIVING buys
    spent_on_buys_usd = sum(s.amount_usd for s in surviving_buys)
    plan.idle_capital_usd = available_usd - spent_on_buys_usd

    # ── Build note ──
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
        "Swap plan: %d sells + %d buys → %d executed (%d quota remaining). "
        "Idle=%.0f USD. %s",
        len(surviving_sells), len(surviving_buys), plan.trades_used,
        plan.remaining_quota, plan.idle_capital_usd, plan.note or "ok",
    )
    return plan


def _find_score(candidates: list, symbol: str) -> float:
    """Find the composite score for a symbol in the candidate list."""
    for c in candidates:
        if c.symbol == symbol:
            return c.composite_score
    return 0.0
