#!/usr/bin/env python3
"""
Orchestrator — 15-minute tick loop wiring all 7 phases together.

Usage:
    python3 main.py                        # live trading (default: 10-min loop)
    python3 main.py --paper-trade          # dry-run mode (fake tx hashes)
    python3 main.py --once                 # run ONE tick then exit (debug)
    python3 main.py --interval 300         # custom tick interval (seconds)

Flow per tick:
    1. Load portfolio state from disk
    2. Run guardrails (drawdown → inactivity → quota)
    3. Handle non-PROCEED verdicts (CIRCUIT_BREAKER, COMPLIANCE_TRADE, SKIP)
    4. If PROCEED: fetch holdings → regime → momentum → portfolio → execute
    5. Record trades, update state, save to disk
    6. Sleep until next tick

Architecture:
    main.py wires together all components. It owns the runtime loop.
    Each component is independently testable and replaceable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# ── Agent imports ──
sys.path.insert(0, str(Path(__file__).resolve().parent))

from execution.guardrails import (
    Verdict, GuardResult,
    run_checks, record_trade, record_compliance_trade, update_peak,
    load_state, save_state, log_trade,
    MAX_TRADES_PER_DAY, DEFAULT_STATE_DIR,
)
from execution.twak_client import (
    TwakClient, TradeResult, Holdings, resolve_address,
)
from strategy.regime import classify_regime, RegimeDecision
from strategy.momentum import discover_candidates
from strategy.portfolio import generate_swap_plan, SwapPlan
from data.cmc_client import cmc_fetch_quotes_prices, cmc_fetch_quotes_momentum
from data.allowlist import is_stablecoin

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("orchestrator")

# ── Constants ──
DEFAULT_INTERVAL_SECONDS = 600      # 10 minutes
REGIME_REFRESH_INTERVAL = 3600      # re-classify regime every hour
PRICE_REFRESH_SECONDS = 300         # refresh CMC prices every 5 min


# ═══════════════════════════════════════════════════════════════════════════
# LLM Client (injectable)
# ═══════════════════════════════════════════════════════════════════════════

class DeepSeekClient:
    """
    Thin wrapper around DeepSeek's OpenAI-compatible API providing the
    .chat(system, user) interface expected by regime.classify_regime().

    DeepSeek API is OpenAI-format, endpoint: https://api.deepseek.com/v1
    Falls back gracefully on any failure — returns neutral regime JSON.
    """

    def __init__(self, model: str = "deepseek-chat"):
        self._model = model
        self._api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self._base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self._client = None

        if not self._api_key:
            log.warning(
                "DEEPSEEK_API_KEY not set — regime classification will "
                "return neutral (no LLM calls possible)"
            )
        else:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                )
                log.debug("LLM: DeepSeek client ready (model=%s)", model)
            except ImportError:
                log.warning(
                    "openai SDK not installed — regime classification will "
                    "return neutral. Install: pip install openai"
                )

    def chat(self, system: str, user: str) -> str:
        """Call DeepSeek. Returns text content, or a neutral JSON on failure."""
        if not self._client:
            return _neutral_fallback("DeepSeek client not configured")

        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                max_tokens=512,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return completion.choices[0].message.content
        except Exception as exc:
            log.error("DeepSeek call failed: %s", exc)
            return _neutral_fallback(f"DeepSeek error: {exc}")


def _neutral_fallback(reason: str) -> str:
    """Return a valid neutral regime JSON when the LLM is unavailable."""
    import json
    return json.dumps({
        "regime": "neutral",
        "confidence": 0.3,
        "reasoning": reason,
        "params": {
            "max_positions": 3,
            "allocation_pct": 0.30,
            "momentum_lookback": "24h",
        },
    })


# ═══════════════════════════════════════════════════════════════════════════
# Paper Trade State Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_holdings_from_state(state: dict) -> Holdings:
    """Construct a Holdings object from state['holdings'] for paper trade mode.

    Unlike the live twak_client.fetch_holdings() which queries the wallet,
    this reconstructs the portfolio from the last saved state so paper trades
    accumulate across ticks — buys add tokens, sells remove them.
    """
    h = Holdings()
    raw = state.get("holdings", {})
    if not raw:
        # First ever tick: seed with simulated $10,000 USDT
        raw = {"USDT": {"balance": 10_000.0, "cost_basis_usd": 10_000.0}}
        state["holdings"] = raw

    h.tokens = {k: dict(v) for k, v in raw.items()}
    # Total value is estimated below with price cache; use cost_basis as fallback
    h.total_value_usd = sum(v.get("cost_basis_usd", 0) for v in h.tokens.values())
    return h


def _apply_paper_holdings(state: dict, plan: SwapPlan) -> None:
    """Update state['holdings'] to reflect executed paper swap plan.

    Mutates state in place. Simulates balance changes for each swap:
    - Sells: remove from_token balance, add to_token (USDT) balance
    - Buys: reduce USDT balance, add to_token balance at execution price
    """
    holdings = state.setdefault("holdings", {})
    if not holdings:
        holdings["USDT"] = {"balance": 10_000.0, "cost_basis_usd": 10_000.0}

    for si in plan.swaps:
        if si.action == "sell":
            # Remove the sold token entirely
            from_tok = holdings.get(si.from_token, {})
            if from_tok:
                from_tok["balance"] = max(0, from_tok.get("balance", 0) - si.amount_token)
                from_tok["cost_basis_usd"] = max(0, from_tok.get("cost_basis_usd", 0) - si.amount_usd)
            # Clean up dust
            if from_tok.get("balance", 0) < 0.0001:
                holdings.pop(si.from_token, None)
            # Add USDT proceeds (with slippage haircut)
            usdt = holdings.setdefault(si.to_token, {"balance": 0, "cost_basis_usd": 0})
            usdt["balance"] += si.amount_usd * 0.985  # 1.5% slippage
            usdt["cost_basis_usd"] = usdt["balance"]  # USDT pegged
            # Clear peak_price for sold token
            holdings.pop(si.from_token + "_peak", None)

        elif si.action == "buy":
            # Reduce USDT balance
            usdt = holdings.get(si.from_token, {})
            if usdt:
                usdt["balance"] = max(0, usdt.get("balance", 0) - si.amount_usd)
                usdt["cost_basis_usd"] = usdt["balance"]
            # Add bought token
            to_tok = holdings.setdefault(si.to_token, {"balance": 0, "cost_basis_usd": 0})
            to_tok["balance"] += si.amount_token
            to_tok["cost_basis_usd"] = to_tok.get("cost_basis_usd", 0) + si.amount_usd
            # Record peak price for trailing stop tracking
            if si.amount_token > 0 and si.amount_usd > 0:
                entry_price = si.amount_usd / si.amount_token
                current_peak = to_tok.get("peak_price", 0)
                if entry_price > current_peak:
                    to_tok["peak_price"] = entry_price
            # Clean up USDT dust
            if usdt.get("balance", 0) < 0.01:
                holdings.pop(si.from_token, None)


def _paper_total_value(state: dict, price_cache: dict[str, float]) -> float:
    """Estimate total portfolio value from paper holdings and current prices."""
    holdings = state.get("holdings", {})
    total = 0.0
    for sym, info in holdings.items():
        balance = info.get("balance", 0)
        if balance <= 0:
            continue
        price = price_cache.get(sym.upper(), price_cache.get(sym, 0))
        if price > 0:
            total += balance * price
        else:
            # Fallback: use cost_basis for stablecoins or if no price available
            total += info.get("cost_basis_usd", 0)
    return total


# ═══════════════════════════════════════════════════════════════════════════
# CMC Price Cache Helper
# ═══════════════════════════════════════════════════════════════════════════

def _build_price_cache(
    holdings,                   # Holdings object from twak.fetch_holdings()
    candidates: list,
) -> dict[str, float]:
    """
    Build a {symbol: usd_price} dict from CMC quotes + cost-basis fallback.

    Always fetches fresh CMC prices for every held volatile token AND every
    candidate. This is critical: the trailing stop-loss compares current price
    against peak. If we used cost-basis as current price, the stop would
    never trigger and the bot would ride a -90% dump to zero.

    Cost-basis prices are used ONLY as a fallback when CMC does not return
    a price for a token (e.g., small-cap, delisted, or API gap).

    Returns a dict keyed by uppercase symbol with float USD prices.
    """
    price: dict[str, float] = {}

    # 1. Collect all symbols needing fresh CMC prices
    symbols_to_fetch: set[str] = set()
    cost_basis_fallbacks: dict[str, float] = {}

    # Held tokens — always fetch CMC prices for volatile ones
    for sym, info in holdings.tokens.items():
        sym_up = sym.upper()
        bal = info.get("balance", 0)
        cost = info.get("cost_basis_usd", 0)
        if bal > 0 and cost > 0:
            cost_basis_fallbacks[sym_up] = cost / bal
            # Fetch fresh prices for everything except pegged stablecoins
            if sym_up not in ("USDT", "FDUSD", "USDC", "DAI", "TUSD", "FRAX", "USDD"):
                symbols_to_fetch.add(sym_up)

    # Candidate tokens not already in the fetch list
    for c in candidates:
        sym_up = c.symbol.upper()
        if sym_up not in symbols_to_fetch and sym_up not in ("USDT", "FDUSD"):
            symbols_to_fetch.add(sym_up)

    # 2. Fetch fresh CMC prices for all needed symbols
    if symbols_to_fetch:
        try:
            fresh = cmc_fetch_quotes_prices(list(symbols_to_fetch))
            for sym, p in fresh.items():
                if p > 0:
                    price[sym.upper()] = p
        except Exception as exc:
            log.warning("CMC price fetch failed: %s — %d symbols unpriced", exc, len(symbols_to_fetch))

    # 3. Fall back to cost-basis ONLY for tokens CMC did not return
    for sym, fallback_p in cost_basis_fallbacks.items():
        if sym not in price:
            price[sym] = fallback_p
            log.debug("Price for %s: using cost-basis fallback ($%.4f)", sym, fallback_p)

    # 4. Stablecoin pegs (always $1.00)
    for stable in ("USDT", "FDUSD", "USDC", "DAI", "TUSD", "FRAX", "USDD"):
        price.setdefault(stable, 1.0)

    log.debug("Price cache: %d tokens (CMC=%d, fallback=%d)",
              len(price),
              sum(1 for k in price if k not in cost_basis_fallbacks or k in price),
              sum(1 for k in price if k in cost_basis_fallbacks and price.get(k) == cost_basis_fallbacks.get(k)))
    return price


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class Orchestrator:
    """
    Main orchestrator — owns the loop, wires all components.

    Lifecycle:
        orch = Orchestrator(twak, llm, mcp_execute, paper_trade=False)
        orch.run(once=False, interval=900)
    """

    def __init__(
        self,
        twak: TwakClient,
        llm: DeepSeekClient,
        mcp_execute: Callable[[str, dict], Any],
        *,
        paper_trade: bool = False,
    ):
        self._twak = twak
        self._llm = llm
        self._mcp = mcp_execute
        self._paper_trade = paper_trade

        # Paper trade uses a separate state file so it never corrupts live data
        self._state_dir = Path(
            os.getenv("AGENT_STATE_DIR", str(DEFAULT_STATE_DIR))
        )
        if paper_trade:
            self._state_dir = self._state_dir / "paper"
        self._state_file = self._state_dir / "portfolio_state.json"

        # Volatile runtime cache (reset each tick)
        self._regime: Optional[RegimeDecision] = None
        self._last_regime_ts: float = 0.0
        self._last_price_ts: float = 0.0
        self._price_cache: dict[str, float] = {}
        self._tick_count: int = 0

        # Ensure state directory exists
        self._state_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "Orchestrator initialized: wallet=%s, paper=%s",
            twak.wallet_address[:10] + "...", paper_trade,
        )

    # ── Public: run loop ─────────────────────────────────────────────────

    def run(self, *, once: bool = False, interval: float = DEFAULT_INTERVAL_SECONDS):
        """Enter the main tick loop. Blocks until interrupted."""
        log.info(
            "Orchestrator starting: %s mode, interval=%ds",
            "once" if once else "loop", interval,
        )
        try:
            while True:
                tick_start = time.monotonic()
                self._tick_count += 1
                log.info("── Tick %d ──", self._tick_count)

                try:
                    self._tick()
                except Exception as exc:
                    log.error(
                        "Unhandled exception in tick %d: %s\n%s",
                        self._tick_count, exc, traceback.format_exc(),
                    )
                    # NEVER crash — log and continue
                    # In production: alert via Telegram/Discord webhook

                if once:
                    log.info("Single tick complete — exiting")
                    return

                elapsed = time.monotonic() - tick_start
                sleep_for = max(0, interval - elapsed)
                log.info(
                    "Tick %d done in %.1fs — sleeping %.0fs",
                    self._tick_count, elapsed, sleep_for,
                )
                time.sleep(sleep_for)
        except KeyboardInterrupt:
            log.info("Orchestrator stopped by user (Ctrl+C)")

    # ── Holdings refresh (every tick, live mode) ──────────────────────────

    def _refresh_holdings(self, state: dict):
        """
        Fetch TWAK portfolio + merge state-tracked invisible tokens.

        TWAK's token registry can't display many competition tokens (GUA,
        SIREN, DEXE, BSB, etc.) — their balances don't appear in ``twak
        wallet portfolio`` output even though the wallet holds them.  This
        method merges TWAK-reported tokens (BNB, USDT) with state-carried
        tokens so the bot always has a complete picture.

        Side effects (intentional):
          - state["holdings"] = merged dict
          - state["current_value_usd"] = total (TWAK + invisible)
          - update_peak(state, total)
        """
        if self._paper_trade:
            return _build_holdings_from_state(state)

        # ── Fetch TWAK portfolio ──────────────────────────────────────
        try:
            latest = self._twak.fetch_holdings()
        except Exception as exc:
            log.error("Holdings fetch failed: %s", exc)
            # Fall back to state-only reconstruction
            return _build_holdings_from_state(state)

        old = state.get("holdings", {})

        # Start with everything TWAK returned
        merged = {sym: dict(info) for sym, info in latest.tokens.items()}
        twak_syms = {s.upper() for s in merged}

        # Carry forward peak_prices for tokens TWAK returned
        for sym in merged:
            old_info = old.get(sym, {})
            stored_peak = old_info.get("peak_price", 0)
            if stored_peak:
                current_val = merged[sym].get("peak_price", 0) or 0
                merged[sym]["peak_price"] = max(stored_peak, current_val)

        # Preserve tokens TWAK did NOT return (invisible tokens)
        for sym, info in old.items():
            if sym.upper() not in twak_syms and sym.upper() != "BNB":
                price = self._price_cache.get(sym.upper(), 0)
                bal = info.get("balance", 0)
                if bal > 0:
                    info = dict(info)
                    if price > 0:
                        info["value_usd"] = bal * price
                    merged[sym] = info

        state["holdings"] = merged

        # Compute total: TWAK total + invisible tokens' value
        total = latest.total_value_usd
        for sym, info in merged.items():
            if sym.upper() not in twak_syms:
                total += info.get("value_usd", info.get("cost_basis_usd", 0))

        update_peak(state, total)
        state["current_value_usd"] = total

        # Build Holdings object for the rest of the tick
        h = Holdings()
        h.tokens = merged
        h.total_value_usd = total
        return h

    # ── Tick implementation ──────────────────────────────────────────────

    def _tick(self):
        """Run one full orchestrator tick."""
        # Step 1 — Load state
        state = load_state(self._state_file)

        # Step 2 — Refresh holdings (always, so manual trades are visible)
        # Fetch TWAK portfolio + merge state-tracked invisible tokens.
        # Updates state["holdings"], state["current_value_usd"], and peak.
        holdings = self._refresh_holdings(state)

        # Step 3 — If TWAK fetch failed fatally and we have no state, abort
        if holdings is None:
            log.error("Holdings unavailable — skipping tick")
            save_state(state, self._state_file)
            return

        total_value = holdings.total_value_usd
        if total_value <= 0:
            log.warning("Portfolio value is $0 — skipping tick (no balance yet?)")
            save_state(state, self._state_file)
            return

        log.info(
            "Portfolio: $%.0f total (%d tokens), peak=$%.0f, drawdown=%.1f%%",
            total_value, len(holdings.tokens),
            state["peak_value_usd"], state["drawdown_pct"],
        )

        # Step 4 — Run guardrails (daily reset, drawdown, inactivity, quota)
        verdict = run_checks(state)

        # Step 5 — SKIP_REBALANCE: quota exhausted, no new positions.
        # BUT still scan trailing stops — a token can crash while quota is dry.
        if verdict.verdict == Verdict.SKIP_REBALANCE:
            if self._paper_trade:
                log.info(
                    "Guardrails: skip_rebalance — quota %d/%d (paper: continuing anyway)",
                    state.get("trades_today", 0), MAX_TRADES_PER_DAY,
                )
            else:
                log.info("Guardrails: %s — %s", verdict.verdict.value, verdict.reason)
                self._tick_exits_only(state, holdings, total_value)
                # _tick_exits_only handles its own save_state
                return
        else:
            log.info("Guardrails: %s — %s", verdict.verdict.value, verdict.reason)

        # Step 6 — Handle heartbeat trade (return after — no full pipeline)
        if verdict.verdict == Verdict.COMPLIANCE_TRADE:
            self._handle_compliance(state)
            save_state(state, self._state_file)
            return

        # Step 7 — Circuit breaker: skip momentum + buys, run exits only
        if verdict.verdict == Verdict.CIRCUIT_BREAKER:
            self._tick_exits_only(state, holdings, total_value)
            save_state(state, self._state_file)
            return

        # Step 8 — PROCEED: full pipeline

        # 8a — Regime classification (once per hour)
        if self._regime is None or (time.monotonic() - self._last_regime_ts) >= REGIME_REFRESH_INTERVAL:
            log.info("Running regime classification...")
            try:
                self._regime = classify_regime(self._llm, self._mcp)
                self._last_regime_ts = time.monotonic()
                state["regime"] = self._regime.regime
                state["regime_updated_ts"] = self._regime.updated_ts
                log.info(
                    "Regime: %s (confidence=%.2f) — %s",
                    self._regime.regime, self._regime.confidence, self._regime.reasoning,
                )
            except Exception as exc:
                log.error("Regime classification crashed: %s — using neutral", exc)
                self._regime = RegimeDecision(error=str(exc))
                state["regime"] = "neutral"
        else:
            stale = time.monotonic() - self._last_regime_ts
            log.debug("Using cached regime (%s, %.0fs stale)", self._regime.regime, stale)

        regime = self._regime.regime if self._regime else "neutral"

        # 8b — Momentum discovery
        log.info("Running momentum discovery (regime=%s)...", regime)
        try:
            # Pass cooldowns from state for penalty box
            cooldowns = state.get("cooldowns", {})
            momentum = discover_candidates(
                mcp_execute=self._mcp,
                regime=regime,
                top_n=2,  # concentrated: 2 positions
                cmc_fetch=cmc_fetch_quotes_momentum,
                cooldowns=cooldowns,
            )
        except Exception as exc:
            log.error("Momentum discovery crashed: %s — skipping tick", exc)
            save_state(state, self._state_file)
            return

        if momentum.error:
            log.warning("Momentum pipeline error: %s", momentum.error)
        if not momentum.candidates:
            log.info("No momentum candidates — HOLD (saving state)")
            save_state(state, self._state_file)
            return

        for i, c in enumerate(momentum.candidates):
            addr = resolve_address(c.symbol)
            addr_str = f"  {addr}" if addr else ""
            log.info("  Candidate #%d: %s%s score=%.3f 1h=+%.1f%% 24h=+%.1f%% vol=$%.0fk (%s)",
                     i+1, c.symbol, addr_str, c.composite_score,
                     c.pct_1h, c.pct_24h, c.vol_24h / 1000, c.reason)

        # 7c — Build price cache
        now_ts = time.monotonic()
        if not self._price_cache or (now_ts - self._last_price_ts) >= PRICE_REFRESH_SECONDS:
            self._price_cache = _build_price_cache(
                holdings, momentum.candidates,
            )
            self._last_price_ts = now_ts

        # 7cp — DEX price override for held tokens (trailing stop MUST use real DEX prices)
        # CMC aggregate price ≠ PancakeSwap pool price. A token can look healthy on
        # CMC while being 50% down on DEX. Override each holding with a TWAK quote.
        if not self._paper_trade:
            for sym, info in holdings.tokens.items():
                bal = info.get("balance", 0)
                if bal <= 0 or is_stablecoin(sym) or sym.upper() == "BNB":
                    continue
                try:
                    quote = self._twak.quote_swap(
                        from_token=sym,
                        to_token="USDT",
                        amount_token=bal,
                        slippage=5,
                    )
                    if quote > 0:
                        dex_price = quote / bal
                        cmc_price = self._price_cache.get(sym.upper(), 0)
                        self._price_cache[sym.upper()] = dex_price
                        if cmc_price > 0 and abs(dex_price - cmc_price) / cmc_price > 0.05:
                            log.info(
                                "  DEX price override %s: CMC=$%.4f → DEX=$%.4f (%.0f%% gap)",
                                sym, cmc_price, dex_price,
                                (dex_price / cmc_price - 1) * 100,
                            )
                except Exception:
                    pass  # DEX quote failed, keep CMC price

        # 7d — Generate swap plan (concentrated: 2 targets, allocation from regime)
        max_pos = self._regime.max_positions if self._regime else 2
        alloc_pct = self._regime.allocation_pct if self._regime else 0.80
        # In paper trade mode, always tell the portfolio layer zero trades used
        # so it never hits its internal quota cap — we want to see every tick.
        effective_trades = 0 if self._paper_trade else state.get("trades_today", 0)
        plan: SwapPlan = generate_swap_plan(
            holdings=holdings.tokens,
            candidates=momentum.candidates,
            price_cache=self._price_cache,
            regime=regime,
            max_positions=max_pos,
            allocation_pct=alloc_pct,
            total_value_usd=total_value,
            trades_today=effective_trades,
        )

        # Merge penalty box cooldowns into state
        if plan.new_cooldowns:
            if "cooldowns" not in state:
                state["cooldowns"] = {}
            state["cooldowns"].update(plan.new_cooldowns)

        self._execute_and_record(state, plan, regime)

    # ── Plan execution ──────────────────────────────────────────────────

    def _execute_and_record(self, state: dict, plan: SwapPlan, regime: str,
                             *, quota_exempt: bool = False):
        """Execute a swap plan via TWAK and record results to state + trade log.

        quota_exempt=True: trades do NOT count toward the 5/day limit.
        Used for circuit breaker exits — risk management must never be
        throttled by the overtrading guard.
        """
        if not plan.swaps:
            log.info("Swap plan empty — nothing to execute")
            state["regime"] = regime
            save_state(state, self._state_file)
            return

        # Log plan summary with addresses so you can track tokens
        for si in plan.swaps:
            addr = resolve_address(si.to_token if si.action == "buy" else si.from_token)
            addr_str = f" ({addr})" if addr else ""
            if si.action == "buy":
                log.info("📈 BUY  $%.0f %s%s → %s  (%s)", si.amount_usd, si.from_token, addr_str, si.to_token, si.reason)
            else:
                log.info("📉 SELL %.0f %s%s → %s  (%s)", si.amount_token, si.from_token, addr_str, si.to_token, si.reason)

        # ── DEX liquidity pre-check: filter buys on shallow pools ──────
        # CMC volume ≠ DEX liquidity. Before executing, quote each buy on
        # the actual DEX and reject if price impact > 30% (DEX quote < 70%
        # of CMC estimate). This prevents draining empty pools like BARD.
        LIQUIDITY_IMPACT_REJECT = 0.30  # reject if DEX offers <70% of CMC value
        valid_swaps = []
        for si in plan.swaps:
            if si.action != "buy":
                valid_swaps.append(si)
                continue
            try:
                quote = self._twak.quote_swap(
                    from_token=si.from_token,
                    to_token=si.to_token,
                    amount_usd=si.amount_usd,
                )
                if quote <= 0:
                    log.warning("  ⚠️ No DEX quote for %s — skipping buy", si.to_token)
                    continue
                # Compare DEX quote to USD spent
                received_usd = quote * self._price_cache.get(
                    si.to_token.upper(),
                    si.amount_usd / max(si.amount_token, 0.0001),
                )
                # For quote_swap returning token count: received_usd = quote * CMC price
                cmc_price = self._price_cache.get(si.to_token.upper(), 0)
                if cmc_price > 0:
                    received_usd = quote * cmc_price
                    ratio = received_usd / max(si.amount_usd, 1)
                else:
                    ratio = 1.0  # no CMC price, let it through
                if ratio < (1.0 - LIQUIDITY_IMPACT_REJECT):
                    log.warning(
                        "  ❌ DEX liquidity check failed: %s buy $%.0f → DEX gives $%.2f "
                        "(%.0f%% impact). Skipping.",
                        si.to_token, si.amount_usd, received_usd, (1 - ratio) * 100,
                    )
                    continue
                log.info(
                    "  ✅ DEX liquidity OK: %s $%.0f → ~$%.2f (%.0f%% impact)",
                    si.to_token, si.amount_usd, received_usd, (1 - ratio) * 100,
                )
            except Exception as exc:
                log.warning("  ⚠️ DEX quote failed for %s: %s — executing anyway", si.to_token, exc)
            valid_swaps.append(si)
        plan.swaps = valid_swaps
        # ── End liquidity pre-check ──────────────────────────────────

        if not plan.swaps:
            log.info("All buys rejected by DEX liquidity check — nothing to execute")
            save_state(state, self._state_file)
            return

        results = self._twak.execute_plan(plan)
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]

        for i, result in enumerate(results):
            if not result.success:
                continue
            state = record_trade(state, result, quota_exempt=quota_exempt)
            swap_reason = ""
            if i < len(plan.swaps):
                swap_reason = plan.swaps[i].reason
            log_trade({
                "ts": datetime.now(timezone.utc).isoformat(),
                "action": "swap",
                "token": result.to_token,
                "from_token": result.from_token,
                "amount": result.amount_token,
                "tx_hash": result.tx_hash,
                "regime": regime,
                "reason": swap_reason or "rebalance",
            })

        # Update holdings in state
        if self._paper_trade:
            _apply_paper_holdings(state, plan)
            total_value = _paper_total_value(state, self._price_cache)
            update_peak(state, total_value)
        else:
            # ── Live mode: TWAK + state-tracked tokens ──────────────
            # TWAK only reports tokens in its symbol registry. Competition
            # tokens like GUA, DEXE, BSB are invisible to TWAK's portfolio
            # display even though the swap succeeds. We must preserve these
            # tokens from state so the bot knows it owns them.
            #
            # Strategy: TWAK tells us BNB + USDT balances (always accurate).
            # State tells us volatile token positions (what we bought/sold).
            # Merge both: TWAK overwrites known tokens, state preserves the
            # tokens TWAK can't see.
            try:
                latest = self._twak.fetch_holdings()
                old = state.get("holdings", {})

                # Start with everything TWAK returned
                merged = {sym: dict(info) for sym, info in latest.tokens.items()}

                # Carry forward peak_prices for tokens TWAK returned
                for sym in merged:
                    old_info = old.get(sym, {})
                    stored_peak = old_info.get("peak_price", 0)
                    if stored_peak:
                        current_val = merged[sym].get("peak_price", 0) or 0
                        merged[sym]["peak_price"] = max(stored_peak, current_val)

                # Preserve tokens TWAK did NOT return (invisible tokens)
                twak_syms = {s.upper() for s in merged}
                for sym, info in old.items():
                    if sym.upper() not in twak_syms and sym.upper() != "BNB":
                        # TWAK didn't see this token — carry it forward
                        # Update its value from the price cache
                        price = self._price_cache.get(sym.upper(), 0)
                        bal = info.get("balance", 0)
                        if price > 0 and bal > 0:
                            info = dict(info)
                            info["value_usd"] = bal * price
                        merged[sym] = info

                # Apply swap plan deltas ONLY for successfully executed swaps.
                # Failed sells must NOT remove the token — it's still in wallet.
                succeeded = {r.from_token.upper() for r in results if r.success and r.from_token}
                succeeded |= {r.to_token.upper() for r in results if r.success and r.to_token}
                for i, si in enumerate(plan.swaps):
                    result = results[i] if i < len(results) else None
                    if not result or not result.success:
                        continue  # skip failed swaps entirely
                    sym = si.to_token if si.action == "buy" else si.from_token
                    if sym.upper() in _STABLES or sym == "BNB":
                        continue
                    if si.action == "buy":
                        # Use actual DEX fill amount parsed from TWAK stdout.
                        # result.output_amount = token count from
                        # "Swapping X USDT -> Y TOKEN via ..." line.
                        # Falls back to CMC estimate if parsing failed.
                        price = self._price_cache.get(sym.upper(), 0)
                        received = result.output_amount
                        if received <= 0:
                            received = si.amount_token  # CMC estimate fallback
                        if received <= 0 and price > 0 and si.amount_usd > 0:
                            received = si.amount_usd / price  # last-resort fallback
                        # Preserve original entry_ts if adding to an existing position
                        existing = old.get(sym) or {}
                        existing_entry = existing.get("entry_ts", "")
                        # Accumulate balance + cost basis
                        prev_bal = float(existing.get("balance", 0))
                        prev_cost = float(existing.get("cost_basis_usd", 0))
                        merged[sym] = {
                            "balance": prev_bal + received,
                            "cost_basis_usd": prev_cost + si.amount_usd,
                            "value_usd": (prev_cost + si.amount_usd),
                            "peak_price": max(
                                price,
                                float(existing.get("peak_price", 0)),
                            ),
                            "entry_ts": existing_entry or datetime.now(timezone.utc).isoformat(),
                        }
                        log.debug(
                            "  State tracking: %s +%.6f tokens (DEX=%.6f, CMC=%.6f)",
                            sym, received, result.output_amount, si.amount_token,
                        )
                    elif si.action == "sell":
                        # Token sold AND tx confirmed — remove from state.
                        # (Failed sells are already skipped by the success guard above;
                        # if we reach here, the chain confirmed the swap.)
                        merged.pop(sym, None)
                        merged.pop(sym.upper(), None)

                state["holdings"] = merged

                # Compute total: TWAK total + invisible tokens' market value
                total = latest.total_value_usd
                for sym, info in merged.items():
                    if sym.upper() not in twak_syms:
                        total += info.get("value_usd",
                                          info.get("cost_basis_usd", 0))
                update_peak(state, total)
            except Exception:
                pass

        # Save final state
        state["regime"] = regime
        save_state(state, self._state_file)

        log.info(
            "Tick %d result: %d/%d swaps succeeded, %d failed",
            self._tick_count, len(successes), len(results), len(failures),
        )
        if failures:
            for f in failures:
                log.warning("  Failed: %s→%s error=%s", f.from_token, f.to_token, f.error)

    def _tick_exits_only(self, state: dict, holdings, total_value: float):
        """
        Circuit breaker tick: trailing stop exits ONLY.

        No new buys. No regular sells (don't sell into a drawdown).
        Only trailing stop-loss exits to protect remaining capital.
        """
        from strategy.portfolio import (
            TRAILING_STOP_PCT, COOLDOWN_SECONDS, SwapInstruction, SwapPlan,
        )

        log.warning(
            "Circuit breaker active — trailing stop scan only "
            "(no buys, no rebalancing sells)"
        )

        # Build price cache for current holdings
        all_symbols = list(holdings.tokens.keys())
        if not all_symbols:
            log.info("No holdings to check trailing stops against")
            save_state(state, self._state_file)
            return

        price_cache = _build_price_cache(holdings, [])

        # Override with DEX prices so trailing stops see real on-chain prices
        if not self._paper_trade:
            for sym, info in holdings.tokens.items():
                bal = info.get("balance", 0)
                if bal <= 0 or is_stablecoin(sym) or sym.upper() == "BNB":
                    continue
                try:
                    quote = self._twak.quote_swap(
                        from_token=sym, to_token="USDT",
                        amount_token=bal, slippage=5,
                    )
                    if quote > 0:
                        dex_price = quote / bal
                        cmc_price = price_cache.get(sym.upper(), 0)
                        price_cache[sym.upper()] = dex_price
                        if cmc_price > 0 and abs(dex_price - cmc_price) / cmc_price > 0.05:
                            log.warning(
                                "  DEX override %s: CMC=$%.4f→DEX=$%.4f (%.0f%% gap)",
                                sym, cmc_price, dex_price,
                                (dex_price / cmc_price - 1) * 100,
                            )
                except Exception:
                    pass

        # Scan each holding for trailing stop violation
        exit_plan = SwapPlan(
            remaining_quota=MAX_TRADES_PER_DAY - state.get("trades_today", 0),
        )
        now = time.time()

        for sym, info in holdings.tokens.items():
            # Skip stablecoins and BNB — trailing stop is for volatile tokens only
            if is_stablecoin(sym) or sym.upper() == "BNB":
                continue

            balance = info.get("balance", 0.0)
            if balance <= 0:
                continue

            current_price = price_cache.get(sym.upper(), price_cache.get(sym, 0))
            if current_price <= 0:
                continue

            stored_peak = info.get("peak_price", 0)
            cost_basis = info.get("cost_basis_usd", current_price)
            if stored_peak > 0:
                peak = stored_peak
            elif cost_basis > 0:
                peak = cost_basis
            else:
                peak = current_price

            # Update peak in state if price went higher
            if current_price > peak:
                if "holdings" not in state:
                    state["holdings"] = {}
                state["holdings"].setdefault(sym, {})
                state["holdings"][sym]["peak_price"] = current_price
                continue

            stop_price = peak * (1.0 - TRAILING_STOP_PCT)
            if current_price <= stop_price:
                amount_usd = balance * current_price
                exit_plan.swaps.append(SwapInstruction(
                    action="sell",
                    from_token=sym,
                    to_token="USDT",
                    amount_usd=amount_usd,
                    amount_token=balance,
                    reason=(
                        f"CIRCUIT_BREAKER_STOP: peak=${peak:.4f} "
                        f"now=${current_price:.4f} "
                        f"({((current_price / peak) - 1) * 100:+.1f}%)"
                    ),
                ))
                # Enforce 2h penalty box
                if "cooldowns" not in state:
                    state["cooldowns"] = {}
                state["cooldowns"][sym.upper()] = now + COOLDOWN_SECONDS
                log.warning(
                    "Circuit breaker: trailing stop on %s peak=$%.4f now=$%.4f "
                    "→ EXIT",
                    sym, peak, current_price,
                )

        if not exit_plan.swaps:
            log.info("Circuit breaker: no trailing stops triggered — HOLD")
            save_state(state, self._state_file)
            return

        log.warning(
            "Circuit breaker: %d trailing stops triggered — executing exits",
            len(exit_plan.swaps),
        )
        self._execute_and_record(state, exit_plan, "risk_off", quota_exempt=True)

    def _handle_compliance(self, state: dict):
        """Execute a $5 USDT → FDUSD compliance trade to stay active."""
        log.info("Compliance trade: $5 USDT → FDUSD (inactivity fallback)")

        from execution.guardrails import MIN_TRADE_USD, COMPLIANCE_FROM, COMPLIANCE_TO
        swap = {
            "action": "buy",
            "from_token": COMPLIANCE_FROM,
            "to_token": COMPLIANCE_TO,
            "amount_token": MIN_TRADE_USD,
            "amount_usd": MIN_TRADE_USD,
            "reason": "Compliance trade: inactivity fallback",
        }

        try:
            result = self._twak.execute_swap(swap)
            if result.success:
                log.info("  ✅ Compliance trade: tx=%s", result.tx_hash)
                state = record_compliance_trade(state, result.tx_hash)
            else:
                log.error("  ❌ Compliance trade failed: %s", result.error)
                # Still mark as recorded to avoid infinite retry loop
                state = record_trade(state, result)
        except Exception as exc:
            log.error("  ❌ Compliance trade exception: %s", exc)
            state = record_trade(state, None)


# ═══════════════════════════════════════════════════════════════════════════
# Emergency Kill Switch
# ═══════════════════════════════════════════════════════════════════════════

_STABLES = {"USDT", "USDC", "DAI", "TUSD", "FDUSD", "FRAX",
            "USDD", "USD1", "USDe", "USDf", "USDF",
            "DUSD", "XUSD", "FRXUSD", "lisUSD", "STABLE"}

def _emergency_sell_all(twak):
    """
    Emergency kill switch: sell every volatile token in the wallet to USDT.

    Does NOT sell BNB (keeps gas) or stablecoins. Executes each sell
    individually with a short delay between. Prints results and exits.

    Merges TWAK portfolio (shows BNB, USDT) with state file (shows GUA,
    SIREN, etc. that TWAK's token registry can't display) so no hidden
    token is left behind.
    """
    import time as _time
    log.warning("🛑 KILL SWITCH ACTIVATED — selling all volatile tokens to USDT")

    # ── Load TWAK holdings (BNB, USDT, maybe others) ──
    try:
        holdings = twak.fetch_holdings()
    except Exception as exc:
        print(f"❌ Failed to fetch wallet: {exc}")
        sys.exit(1)

    tokens = dict(holdings.tokens)  # copy so we can mutate

    # ── Merge state-file holdings for tokens TWAK can't see ──
    state = load_state()
    state_holdings = state.get("holdings", {})
    twak_upper = {s.upper() for s in tokens}
    merged_any = False
    for sym, info in state_holdings.items():
        sym_up = sym.upper()
        if sym_up in twak_upper or sym_up in _STABLES or sym_up == "BNB":
            continue
        bal = float(info.get("balance", 0))
        if bal <= 0:
            continue
        tokens[sym_up] = dict(info)
        tokens[sym_up]["balance"] = bal
        tokens[sym_up]["value_usd"] = float(info.get("value_usd", info.get("cost_basis_usd", 0)))
        merged_any = True
        log.info("KILL: merged hidden token %s (%.4f) from state file", sym_up, bal)

    if merged_any:
        log.warning("KILL: %d hidden tokens added — double-check balances on BSCscan!", sum(
            1 for s in tokens if s.upper() not in twak_upper
        ))

    sold, failed, skipped = 0, 0, 0

    for sym, info in tokens.items():
        sym_up = sym.upper()
        balance = float(info.get("balance", 0))

        # Skip stablecoins and BNB gas
        if sym_up in _STABLES or sym_up == "BNB":
            skipped += 1
            continue

        if balance <= 0:
            skipped += 1
            continue

        log.warning("KILL: selling %.6f %s → USDT", balance, sym)
        swap = {
            "action": "sell",
            "from_token": sym,
            "to_token": "USDT",
            "amount_token": balance,
            "amount_usd": float(info.get("value_usd", 0)),
            "reason": "KILL SWITCH: emergency sell-all",
        }

        try:
            result = twak.execute_swap(swap)
            if result.success:
                log.warning("  ✅ SOLD %s: tx=%s", sym, result.tx_hash)
                sold += 1
            else:
                log.error("  ❌ FAILED %s: %s", sym, result.error)
                failed += 1
        except Exception as exc:
            log.error("  ❌ EXCEPTION selling %s: %s", sym, exc)
            failed += 1

        remaining = sum(1 for s in tokens if s.upper() not in _STABLES and s.upper() != "BNB"
                        and float(tokens[s].get("balance", 0)) > 0)
        if sold + failed < remaining:
            _time.sleep(8)

    # ── Refresh holdings and update state file ──
    try:
        final_holdings = twak.fetch_holdings()
        state["holdings"] = dict(final_holdings.tokens)
        state["current_value_usd"] = final_holdings.total_value_usd
        state["peak_value_usd"] = max(
            state.get("peak_value_usd", 0), final_holdings.total_value_usd
        )
        state["drawdown_pct"] = (
            0.0 if state["peak_value_usd"] <= 0
            else 1.0 - state["current_value_usd"] / state["peak_value_usd"]
        )
        state["emergency_triggered"] = True
        save_state(state)
        print("✅ State file updated")
    except Exception as exc:
        print(f"⚠️  Could not update state file: {exc}")

    print(f"\n🛑 Kill switch complete: {sold} sold, {failed} failed, {skipped} skipped")
    if sold > 0:
        print("✅ Remaining balance should be USDT + BNB. Check with:")
        print("   twak wallet portfolio --chains bsc --json")


def _update_state_from_wallet(twak):
    """
    Fetch TWAK portfolio, merge with state-tracked invisible tokens,
    and save to portfolio_state.json.  Use after manual TWAK swaps so
    the bot's next tick reflects your trades.
    """
    print("📡 Fetching wallet from TWAK...")
    try:
        latest = twak.fetch_holdings()
    except Exception as exc:
        print(f"❌ Failed: {exc}")
        sys.exit(1)

    state = load_state()
    old = state.get("holdings", {})

    merged = {sym: dict(info) for sym, info in latest.tokens.items()}
    twak_syms = {s.upper() for s in merged}

    # Carry forward peak_prices
    for sym in merged:
        old_info = old.get(sym, {})
        stored_peak = old_info.get("peak_price", 0)
        if stored_peak:
            current_val = merged[sym].get("peak_price", 0) or 0
            merged[sym]["peak_price"] = max(stored_peak, current_val)

    # Preserve invisible tokens
    for sym, info in old.items():
        if sym.upper() not in twak_syms and sym.upper() != "BNB":
            bal = info.get("balance", 0)
            if bal > 0:
                merged[sym] = dict(info)

    state["holdings"] = merged

    total = latest.total_value_usd
    for sym, info in merged.items():
        if sym.upper() not in twak_syms:
            total += info.get("value_usd", info.get("cost_basis_usd", 0))

    state["current_value_usd"] = total
    state["peak_value_usd"] = max(state.get("peak_value_usd", 0), total)
    state["drawdown_pct"] = (
        0.0 if state["peak_value_usd"] <= 0
        else 1.0 - state["current_value_usd"] / state["peak_value_usd"]
    )

    save_state(state)
    print(f"✅ State updated: {len(merged)} tokens, ${total:.2f} total")
    for sym, info in merged.items():
        bal = info.get("balance", 0)
        val = info.get("value_usd", info.get("cost_basis_usd", "?"))
        print(f"   {sym}: {bal:.6f} (~${val:.2f})")
    print(f"   peak=${state['peak_value_usd']:.2f}, drawdown={state['drawdown_pct']:.1%}")


def _remove_holding_from_state(symbol: str):
    """Remove a single token from state/portfolio_state.json."""
    sym = symbol.upper()
    state = load_state()
    old = state.get("holdings", {})

    if sym not in old:
        # Also try case-insensitive
        match = next((k for k in old if k.upper() == sym), None)
        if match:
            sym = match
        else:
            print(f"⚠️  {symbol} not found in holdings: {list(old.keys())}")
            sys.exit(1)

    removed = old.pop(sym)
    bal = removed.get("balance", 0)
    val = removed.get("value_usd", removed.get("cost_basis_usd", 0))
    print(f"🗑️  Removed {sym}: balance={bal}, value=${val:.2f}")

    state["holdings"] = old
    save_state(state)
    print(f"✅ State saved — now holding: {list(old.keys())}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BNB Hack AI Trading Agent — Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py                         live trading, 10-min loop
  python3 main.py --paper-trade           dry-run (fake transactions)
  python3 main.py --once --paper-trade    single dry-run tick
  python3 main.py --interval 300          custom 5-min interval
  python3 main.py --kill-switch           emergency: sell EVERYTHING to USDT
  python3 main.py --update-state          sync state after manual trades
        """,
    )
    parser.add_argument(
        "--paper-trade", action="store_true",
        help="Dry-run mode — generates fake tx hashes, no real swaps",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single tick then exit (debug/test)",
    )
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_INTERVAL_SECONDS,
        help=f"Seconds between ticks (default: {DEFAULT_INTERVAL_SECONDS}, 10 min)",
    )
    parser.add_argument(
        "--kill-switch", action="store_true",
        help="EMERGENCY: sell all volatile tokens to USDT immediately and exit",
    )
    parser.add_argument(
        "--update-state", action="store_true",
        help="Fetch TWAK portfolio + merge invisible tokens → save state, then exit",
    )
    parser.add_argument(
        "--remove-holding", type=str, metavar="SYMBOL",
        help="Remove a token from state (use after manually selling a token TWAK can't see)",
    )
    parser.add_argument(
        "--twak-bin", default="twak",
        help="Path to TWAK CLI binary (default: twak)",
    )
    args = parser.parse_args()

    # ── Validate environment ──────────────────────────────────────────
    missing: list[str] = []
    # In paper trade mode, wallet credentials are optional (no real txs)
    if not args.paper_trade:
        if not os.getenv("WALLET_PASSWORD"):
            missing.append("WALLET_PASSWORD")
        if not os.getenv("PRIVATE_KEY") and not os.getenv("WALLET_ADDRESS"):
            from bnbagent import EVMWalletProvider
            if not EVMWalletProvider.keystore_exists():
                missing.append("PRIVATE_KEY or WALLET_ADDRESS")

    if missing:
        print(f"❌ Missing required env vars: {', '.join(missing)}")
        print("   Set them in .env or export them before running.")
        sys.exit(1)

    # ── Bootstrap components ──────────────────────────────────────────
    log.debug("Bootstrapping orchestrator...")

    llm = DeepSeekClient()
    twak = TwakClient.from_env(twak_bin=args.twak_bin, paper_trade=args.paper_trade)
    log.debug("Wallet: %s (paper=%s)", twak.wallet_address, args.paper_trade)

    # ── Kill switch: sell everything to USDT ─────────────────────────────
    if args.kill_switch:
        if twak._paper_trade:
            print("❌ Kill switch requires live mode (no --paper-trade)")
            sys.exit(1)
        _emergency_sell_all(twak)
        return

    # ── Update state: fetch + merge + save (for after manual trades) ──
    if args.update_state:
        if twak._paper_trade:
            print("❌ --update-state requires live mode (no --paper-trade)")
            sys.exit(1)
        _update_state_from_wallet(twak)
        return

    # ── Remove holding (for after manually selling an invisible token) ──
    if args.remove_holding:
        _remove_holding_from_state(args.remove_holding)
        return

    # MCP executor: the orchestrator expects a callable.
    # For live runs, this is the CMC MCP tool from the skill hub.
    # For paper/dry runs without MCP access, we use a minimal stub.
    mcp = _build_mcp_executor(args.paper_trade)

    orch = Orchestrator(twak, llm, mcp, paper_trade=args.paper_trade)
    orch.run(once=args.once, interval=args.interval)


def _build_mcp_executor(paper_trade: bool) -> Callable:
    """
    Build the MCP executor callable used by regime + momentum.

    In live mode: delegates to the cmc-skill-hub MCP tools.
    In paper trade: provides a stub that returns empty/safe data.
    """
    # Try the real CMC bridge first (works in both live and paper trade).
    # Only fall back to a stub when CMC_API_KEY is not configured at all.
    try:
        from data.cmc_client import cmc_mcp_bridge
        log.debug("MCP: CMC bridge ready (real market data)")
        return cmc_mcp_bridge
    except ImportError:
        log.warning(
            "CMC MCP bridge not available. Regime + momentum will fall "
            "back to empty data if CMC_API_KEY is not set."
        )

    # Last resort: empty stub (regime → neutral, momentum → none)
    def _stub(name: str, params: dict) -> dict:
        log.warning("MCP stub: no real data source for %s", name)
        return {"ok": True, "data": {}}
    return _stub


if __name__ == "__main__":
    main()
