#!/usr/bin/env python3
"""
Orchestrator — 15-minute tick loop wiring all 7 phases together.

Usage:
    python3 main.py                        # live trading (default: 15-min loop)
    python3 main.py --paper-trade          # dry-run mode (fake tx hashes)
    python3 main.py --once                 # run ONE tick then exit (debug)
    python3 main.py --interval 300         # custom tick interval (seconds)

Flow per tick:
    1. Load portfolio state from disk
    2. Run guardrails (drawdown → inactivity → quota)
    3. Handle non-PROCEED verdicts (EMERGENCY_SELL, COMPLIANCE_TRADE, SKIP)
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
    TwakClient, TradeResult,
)
from strategy.regime import classify_regime, RegimeDecision
from strategy.momentum import discover_candidates
from strategy.portfolio import generate_swap_plan, SwapPlan
from data.cmc_client import cmc_fetch_quotes_prices, cmc_fetch_quotes_momentum

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("orchestrator")

# ── Constants ──
DEFAULT_INTERVAL_SECONDS = 900      # 15 minutes
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
                log.info("LLM: DeepSeek client ready (model=%s)", model)
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
# CMC Price Cache Helper
# ═══════════════════════════════════════════════════════════════════════════

def _build_price_cache(
    holdings,                   # Holdings object from twak.fetch_holdings()
    candidates: list,
) -> dict[str, float]:
    """
    Build a {symbol: usd_price} dict from holdings + CMC k-line data.

    Holdings provide accurate cost-basis prices for owned tokens.
    CMC k-line provides current market prices for candidate tokens.

    Returns a dict keyed by uppercase symbol with float USD prices.
    """
    price: dict[str, float] = {}

    # 1. Prices from holdings (cost_basis_usd / balance)
    for sym, info in holdings.tokens.items():
        sym_up = sym.upper()
        bal = info.get("balance", 0)
        cost = info.get("cost_basis_usd", 0)
        if bal > 0 and cost > 0:
            price[sym_up] = cost / bal

    # 2. CMC k-line prices for candidate tokens not already priced
    unpriced = [c.symbol for c in candidates if c.symbol.upper() not in price]
    # Also add USDT and BNB for reference
    for ref in ["USDT", "BNB"]:
        if ref not in price and ref not in [u.upper() for u in unpriced]:
            unpriced.append(ref)

    if unpriced:
        try:
            klines = cmc_fetch_quotes_prices(unpriced, interval="15m", count=1)
            for sym, p in klines.items():
                if p > 0:
                    price[sym.upper()] = p
        except Exception as exc:
            log.warning("CMC k-line price fetch failed: %s", exc)

    # 3. Fallbacks for essentials
    price.setdefault("USDT", 1.0)
    price.setdefault("FDUSD", 1.0)

    log.debug("Price cache: %d tokens priced", len(price))
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

        # Volatile runtime cache (reset each tick)
        self._regime: Optional[RegimeDecision] = None
        self._last_regime_ts: float = 0.0
        self._last_price_ts: float = 0.0
        self._price_cache: dict[str, float] = {}
        self._tick_count: int = 0

        # Ensure state directory exists
        DEFAULT_STATE_DIR.mkdir(parents=True, exist_ok=True)

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

    # ── Tick implementation ──────────────────────────────────────────────

    def _tick(self):
        """Run one full orchestrator tick."""
        # Step 1 — Load state
        state = load_state()

        # Step 2 — Run guardrails (daily reset, drawdown, inactivity, quota)
        verdict = run_checks(state)
        log.info("Guardrails: %s — %s", verdict.verdict.value, verdict.reason)

        # Step 3 — Handle non-PROCEED verdicts
        if verdict.verdict == Verdict.EMERGENCY_SELL:
            self._handle_emergency(state, verdict)
            save_state(state)
            return

        if verdict.verdict == Verdict.SKIP_REBALANCE:
            # Just save state (date reset may have mutated it) and sleep
            save_state(state)
            return

        if verdict.verdict == Verdict.COMPLIANCE_TRADE:
            self._handle_compliance(state)
            save_state(state)
            return

        # Step 4 — PROCEED: full pipeline
        assert verdict.verdict == Verdict.PROCEED

        # 4a — Fetch holdings + build price cache
        try:
            holdings = self._twak.fetch_holdings()
        except Exception as exc:
            log.error("Holdings fetch failed: %s — skipping tick", exc)
            save_state(state)
            return

        total_value = holdings.total_value_usd
        if total_value <= 0:
            log.warning("Portfolio value is $0 — skipping tick (no balance yet?)")
            update_peak(state, 0)
            save_state(state)
            return

        # Update peak tracking
        state = update_peak(state, total_value)
        log.info(
            "Portfolio: $%.0f total, peak=$%.0f, drawdown=%.1f%%",
            total_value, state["peak_value_usd"], state["drawdown_pct"],
        )

        # 4b — Regime classification (once per hour)
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

        # 4c — Momentum discovery
        log.info("Running momentum discovery (regime=%s)...", regime)
        try:
            momentum = discover_candidates(
                mcp_execute=self._mcp,
                regime=regime,
                top_n=5,
                cmc_fetch=cmc_fetch_quotes_momentum,
            )
        except Exception as exc:
            log.error("Momentum discovery crashed: %s — skipping tick", exc)
            save_state(state)
            return

        if momentum.error:
            log.warning("Momentum pipeline error: %s", momentum.error)
        if not momentum.candidates:
            log.info("No momentum candidates — HOLD (saving state)")
            save_state(state)
            return

        for i, c in enumerate(momentum.candidates):
            log.info("  Candidate #%d: %s score=%.3f (%s)", i+1, c.symbol, c.composite_score, c.reason)

        # 4d — Build price cache (reuse holdings from step 4a, no extra TWAK call)
        now = time.monotonic()
        if not self._price_cache or (now - self._last_price_ts) >= PRICE_REFRESH_SECONDS:
            self._price_cache = _build_price_cache(
                holdings, momentum.candidates,
            )
            self._last_price_ts = now

        # 4e — Generate swap plan
        plan: SwapPlan = generate_swap_plan(
            holdings=holdings.tokens,
            candidates=momentum.candidates,
            price_cache=self._price_cache,
            regime=regime,
            max_positions=self._regime.max_positions if self._regime else 3,
            allocation_pct=self._regime.allocation_pct if self._regime else 0.30,
            total_value_usd=total_value,
            trades_today=state.get("trades_today", 0),
        )

        log.info("Swap plan: %d swaps, %d quota remaining, $%.0f idle",
                 plan.trades_used, plan.remaining_quota, plan.idle_capital_usd)
        if plan.note:
            log.info("  Note: %s", plan.note)

        if not plan.swaps:
            log.info("No swaps needed — HOLD")
            save_state(state)
            return

        # 4f — Execute plan
        log.info("Executing %d swaps...", len(plan.swaps))
        results = self._twak.execute_plan(plan)
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]

        # 4g — Record trades (successful only — failures don't consume quota)
        for result in results:
            if not result.success:
                continue
            state = record_trade(state, result)
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "action": "swap",
                "token": result.to_token,
                "from_token": result.from_token,
                "amount": result.amount_token,
                "tx_hash": result.tx_hash,
                "regime": regime,
                "reason": "momentum rebalance",
            }
            log_trade(entry)

        # Update holdings in state for emergency-sell awareness
        try:
            latest = self._twak.fetch_holdings()
            state["holdings"] = latest.tokens
            update_peak(state, latest.total_value_usd)
        except Exception:
            # Non-fatal: holdings snapshot is best-effort for state tracking
            pass

        # Save final state
        state["regime"] = regime
        save_state(state)

        log.info(
            "Tick %d result: %d/%d swaps succeeded, %d failed",
            self._tick_count, len(successes), len(results), len(failures),
        )
        if failures:
            for f in failures:
                log.warning("  Failed: %s→%s error=%s", f.from_token, f.to_token, f.error)

    # ── Verdict handlers ─────────────────────────────────────────────────

    def _handle_emergency(self, state: dict, verdict: GuardResult):
        """Execute emergency sell plan: dump all volatile → USDT."""
        log.critical(
            "🚨 EMERGENCY SELL: %s — executing %d swaps",
            verdict.reason, len(verdict.emergency_swaps),
        )
        if not verdict.emergency_swaps:
            log.warning("Emergency triggered but no volatile tokens to sell")
            save_state(state)
            return

        for i, swap in enumerate(verdict.emergency_swaps):
            log.info(
                "  Emergency swap %d/%d: %s → USDT (%.0f tokens)",
                i + 1, len(verdict.emergency_swaps),
                swap["from_token"], swap["amount_token"],
            )
            try:
                # Emergency swaps bypass normal plan execution — direct TWAK
                result = self._twak.execute_swap(swap)
                if result.success:
                    log.info("    ✅ tx=%s", result.tx_hash)
                    state = record_trade(state, result)
                    log_trade({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "action": "emergency_sell",
                        "token": swap["to_token"],
                        "from_token": swap["from_token"],
                        "amount": swap["amount_token"],
                        "amount_usd": swap["amount_usd"],
                        "tx_hash": result.tx_hash,
                        "regime": "risk_off",
                        "reason": swap["reason"],
                    })
                else:
                    log.error("    ❌ FAILED: %s", result.error)
            except Exception as exc:
                log.error("    ❌ EXCEPTION: %s", exc)

        log.critical("🚨 Emergency exit complete — agent PAUSED until manual reset")

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
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BNB Hack AI Trading Agent — Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py                         live trading, 15-min loop
  python3 main.py --paper-trade           dry-run (fake transactions)
  python3 main.py --once --paper-trade    single dry-run tick
  python3 main.py --interval 300          custom 5-min interval
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
        help=f"Seconds between ticks (default: {DEFAULT_INTERVAL_SECONDS}, 15 min)",
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
    log.info("Bootstrapping orchestrator...")

    llm = DeepSeekClient()
    twak = TwakClient.from_env(twak_bin=args.twak_bin, paper_trade=args.paper_trade)
    log.info("Wallet: %s (paper=%s)", twak.wallet_address, args.paper_trade)

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
        log.info("MCP: CMC bridge ready (real market data)")
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
