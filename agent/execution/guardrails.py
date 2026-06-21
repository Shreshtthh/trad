"""
Guardrails — risk limits, drawdown brake, inactivity fallback, state persistence.

Called at the top of every orchestrator tick (15-min loop) BEFORE momentum
scan or portfolio rebalancing. All checks are rule-based — zero LLM on the
hot path.

Order of checks (fast-fail):
  1. Drawdown monitor: if current_value < 0.75 × peak → EMERGENCY_SELL
  2. Trade counter reset: if UTC date changed → trades_today = 0
  3. Inactivity fallback: if >20h since last_trade_ts → COMPLIANCE_TRADE
  4. Quota check: if trades_today >= 5 → SKIP_REBALANCE
  5. Otherwise → PROCEED to momentum/portfolio/execution
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Constants ──
MAX_DRAWDOWN_PCT = 0.25          # 25% from peak → emergency
INACTIVITY_HOURS = 20             # force compliance trade after 20h idle
MAX_TRADES_PER_DAY = 5
MIN_TRADE_USD = 5.0               # compliance trade size
COMPLIANCE_FROM = "USDT"          # compliance swap from
COMPLIANCE_TO = "FDUSD"           # compliance swap to
SLIPPAGE_DEFAULT = 0.01           # 1% for PancakeSwap V2

# ── State path ──
DEFAULT_STATE_DIR = Path(os.getenv("AGENT_STATE_DIR", str(Path(__file__).resolve().parent.parent.parent / "state")))


class Verdict(Enum):
    """What the orchestrator should do this tick."""
    PROCEED = "proceed"                    # all clear → momentum → portfolio → execute
    SKIP_REBALANCE = "skip_rebalance"      # quota exhausted or nothing to do
    COMPLIANCE_TRADE = "compliance_trade"   # >20h idle → force $5 USDT→FDUSD
    EMERGENCY_SELL = "emergency_sell"      # ≥25% drawdown → sell all volatile, pause


@dataclass
class GuardResult:
    verdict: Verdict
    reason: str = ""
    # For EMERGENCY_SELL: list of SwapInstruction-like dicts to execute
    emergency_swaps: list[dict] = field(default_factory=list)

    @property
    def is_blocking(self) -> bool:
        """True if the orchestrator should not proceed to momentum/portfolio."""
        return self.verdict in (Verdict.SKIP_REBALANCE, Verdict.EMERGENCY_SELL)


# ── State I/O ────────────────────────────────────────────────────────────

def load_state(path: Path | None = None) -> dict:
    """Load portfolio state from disk. Returns defaults if file missing."""
    p = path or DEFAULT_STATE_DIR / "portfolio_state.json"
    if not p.exists():
        log.info("No state file at %s — starting with defaults", p)
        return _default_state()
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.error("Failed to read state file %s: %s — using defaults", p, exc)
        return _default_state()


def save_state(state: dict, path: Path | None = None) -> None:
    """Write portfolio state to disk (atomic write)."""
    p = path or DEFAULT_STATE_DIR / "portfolio_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, p)


def _default_state() -> dict:
    return {
        "peak_value_usd": 0.0,
        "current_value_usd": 0.0,
        "drawdown_pct": 0.0,
        "emergency_triggered": False,
        "trades_today": 0,
        "last_trade_date": "",         # "2026-06-23" — for midnight reset
        "last_trade_ts": "",           # ISO 8601
        "holdings": {},
        "regime": "neutral",
        "regime_updated_ts": "",
    }


# ── Trade log (append-only) ──────────────────────────────────────────────

def log_trade(entry: dict, path: Path | None = None) -> None:
    """Append a trade entry to trade_log.json."""
    p = path or DEFAULT_STATE_DIR / "trade_log.json"
    p.parent.mkdir(parents=True, exist_ok=True)

    log_data: dict[str, list] = {"trades": []}
    if p.exists():
        try:
            with open(p) as f:
                log_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            log.warning("trade_log.json corrupt — starting fresh")

    log_data.setdefault("trades", []).append(entry)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    os.replace(tmp, p)

    log.info("Trade logged: %s %s %.6f %s tx=%s",
             entry.get("action"), entry.get("token"),
             entry.get("amount", 0), entry.get("reason", ""),
             entry.get("tx_hash", "?"))


# ── Guardrail checks ─────────────────────────────────────────────────────

def check_daily_reset(state: dict) -> dict:
    """Reset trades_today if UTC date changed since last_trade_date."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    last_date = state.get("last_trade_date", "")
    if last_date != today:
        state["trades_today"] = 0
        state["last_trade_date"] = today
        log.info("New UTC day (%s) — trades_today reset to 0", today)
    return state


def check_drawdown(state: dict) -> GuardResult:
    """
    Check if drawdown exceeds 25% from peak.

    Compares current_value_usd against peak_value_usd. If the portfolio has
    lost ≥25% from its all-time high, return EMERGENCY_SELL with a plan to
    sell every volatile token to USDT.

    Emergency is LATCHING: once triggered, stays true even if portfolio
    recovers. Only a manual reset (editing state file) clears it.
    """
    if state.get("emergency_triggered"):
        return GuardResult(
            verdict=Verdict.EMERGENCY_SELL,
            reason="EMERGENCY already triggered — paused until manual reset",
        )

    peak = state.get("peak_value_usd", 0)
    current = state.get("current_value_usd", 0)

    if peak <= 0:
        # First run — set peak to current
        if current > 0:
            state["peak_value_usd"] = current
            state["drawdown_pct"] = 0.0
        return GuardResult(verdict=Verdict.PROCEED, reason="Initial peak set")

    drawdown = (peak - current) / peak if peak > 0 else 0
    state["drawdown_pct"] = round(drawdown * 100, 2)

    if drawdown >= MAX_DRAWDOWN_PCT:
        state["emergency_triggered"] = True
        log.error(
            "DRAWDOWN EMERGENCY: peak=$%.0f current=$%.0f drawdown=%.1f%% (limit=%.0f%%)",
            peak, current, drawdown * 100, MAX_DRAWDOWN_PCT * 100,
        )

        # Build emergency sell list: every volatile holding → USDT
        emergency_swaps: list[dict] = []
        from data.allowlist import is_stablecoin
        for sym, info in state.get("holdings", {}).items():
            if is_stablecoin(sym) or sym == "BNB":
                continue  # keep stables + BNB (gas)
            balance = info.get("balance", 0)
            if balance > 0:
                emergency_swaps.append({
                    "action": "sell",
                    "from_token": sym,
                    "to_token": "USDT",
                    "amount_token": balance,
                    "amount_usd": info.get("cost_basis_usd", 0),
                    "reason": f"EMERGENCY: {drawdown*100:.1f}% drawdown",
                })

        return GuardResult(
            verdict=Verdict.EMERGENCY_SELL,
            reason=f"Emergency: {drawdown*100:.1f}% drawdown ≥ {MAX_DRAWDOWN_PCT*100:.0f}%",
            emergency_swaps=emergency_swaps,
        )

    return GuardResult(verdict=Verdict.PROCEED, reason=f"Drawdown {drawdown*100:.1f}% OK")


def check_inactivity(state: dict) -> GuardResult | None:
    """
    Check if >20h since last trade. Returns COMPLIANCE_TRADE if idle.

    Competition rule: at least 1 trade per day. If no trade in 20 hours,
    force a $5 USDT → FDUSD swap to stay compliant.

    Returns None when no action needed (caller proceeds to normal flow).
    """
    last_ts = state.get("last_trade_ts", "")
    if not last_ts:
        # No trades ever — this is the first run. Let the bot do a real
        # momentum scan. Don't force a compliance trade on tick 1.
        return None

    try:
        last = datetime.fromisoformat(last_ts)
    except (ValueError, TypeError):
        log.warning("Unparseable last_trade_ts=%r — forcing compliance trade", last_ts)
        return GuardResult(
            verdict=Verdict.COMPLIANCE_TRADE,
            reason="Unparseable last_trade_ts",
        )

    elapsed = datetime.now(timezone.utc) - last
    hours = elapsed.total_seconds() / 3600

    if hours >= INACTIVITY_HOURS:
        return GuardResult(
            verdict=Verdict.COMPLIANCE_TRADE,
            reason=f"Inactive {hours:.1f}h ≥ {INACTIVITY_HOURS}h — compliance trade",
        )

    return None  # all clear


def check_quota(state: dict) -> GuardResult:
    """Check if daily trade quota is exhausted."""
    used = state.get("trades_today", 0)
    if used >= MAX_TRADES_PER_DAY:
        return GuardResult(
            verdict=Verdict.SKIP_REBALANCE,
            reason=f"Daily quota exhausted ({used}/{MAX_TRADES_PER_DAY})",
        )
    remaining = MAX_TRADES_PER_DAY - used
    return GuardResult(
        verdict=Verdict.PROCEED,
        reason=f"Quota OK ({used}/{MAX_TRADES_PER_DAY}, {remaining} remaining)",
    )


# ── State mutation helpers (called by orchestrator after execution) ──────

def record_trade(state: dict, trade_result) -> dict:
    """
    Update state after a successful trade.

    Increments trades_today, updates last_trade_ts, and bumps peak if needed.
    Mutates state in place and returns it.
    """
    state["trades_today"] = state.get("trades_today", 0) + 1
    state["last_trade_ts"] = datetime.now(timezone.utc).isoformat()
    state["last_trade_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    current = state.get("current_value_usd", 0)
    if current > state.get("peak_value_usd", 0):
        state["peak_value_usd"] = current
        log.info("New peak: $%.0f", current)

    return state


def record_compliance_trade(state: dict, tx_hash: str) -> dict:
    """Record a compliance trade and log it."""
    state = record_trade(state, None)
    log_trade({
        "ts": state["last_trade_ts"],
        "action": "buy",
        "token": COMPLIANCE_TO,
        "from_token": COMPLIANCE_FROM,
        "amount": MIN_TRADE_USD / 1.0,  # ~5 FDUSD at $1
        "amount_usd": MIN_TRADE_USD,
        "tx_hash": tx_hash,
        "regime": state.get("regime", "neutral"),
        "reason": f"Compliance trade: {INACTIVITY_HOURS}h inactivity fallback",
    })
    return state


def update_peak(state: dict, current_value_usd: float) -> dict:
    """Update peak if current value exceeds it. Call after portfolio valuation."""
    if current_value_usd > state.get("peak_value_usd", 0):
        state["peak_value_usd"] = current_value_usd
        state["drawdown_pct"] = 0.0
    elif state.get("peak_value_usd", 0) > 0:
        peak = state["peak_value_usd"]
        state["drawdown_pct"] = round((peak - current_value_usd) / peak * 100, 2)
    state["current_value_usd"] = current_value_usd
    return state


# ── Full pre-tick check (single call for orchestrator) ───────────────────

def run_checks(state: dict) -> GuardResult:
    """
    Run all pre-tick guardrail checks in priority order.

    Call from the orchestrator loop before momentum/portfolio steps.
    Returns the first actionable or blocking verdict.

    Priority:
      1. Daily trade counter reset (mutates state, not a verdict)
      2. Drawdown check (EMERGENCY_SELL blocks everything)
      3. Inactivity check (COMPLIANCE_TRADE skips normal flow)
      4. Quota check (SKIP_REBALANCE if no trades left)
      5. PROCEED otherwise
    """
    # Step 1 — reset counter if new day
    state = check_daily_reset(state)

    # Step 2 — drawdown
    dd = check_drawdown(state)
    if dd.verdict != Verdict.PROCEED:
        return dd

    # Step 3 — inactivity
    ia = check_inactivity(state)
    if ia is not None:
        return ia

    # Step 4 — quota
    return check_quota(state)
