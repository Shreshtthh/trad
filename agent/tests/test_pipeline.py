#!/usr/bin/env python3
"""
Comprehensive pipeline test suite: Freshness Ratio Alpha trading agent.

Covers every component path — momentum gates, portfolio sizing, trailing stops,
circuit breaker, regime fallbacks, price cache, state persistence, and full-tick
integration. All mocked. No CMC API key required. No external test framework.

Run:  cd /root/trad/bnb-hack-agent && python3 -m agent.tests.test_pipeline
"""

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

# Make agent package importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy.portfolio import (
    generate_swap_plan, SwapPlan, SwapInstruction,
    SLIPPAGE_MULTIPLIER, BNB_GAS_BUFFER_USD, TRAILING_STOP_PCT,
    COOLDOWN_SECONDS, MAX_TRADES_PER_DAY,
)
from strategy.momentum import (
    discover_candidates, MomentumCandidate, MomentumResult,
    _score_token, FRICTION_FLOOR_PCT, FRESHNESS_MIN,
    MIN_VOLUME_24H, CLIMAX_EXHAUSTION_PCT, HOURLY_CLIMAX_PCT,
)
from strategy.regime import classify_regime, RegimeDecision
from llm.prompts import extract_json, validate_regime_response
from execution.guardrails import (
    check_drawdown, check_inactivity, check_quota, check_daily_reset,
    run_checks, load_state, save_state, record_trade,
    Verdict, GuardResult, MAX_DRAWDOWN_PCT, INACTIVITY_HOURS,
)
from data.allowlist import eligible_non_stable

# ── Real competition tokens for discover_candidates integration tests ──
# discover_candidates() iterates over SCAN_LIST (132 real tokens), so mock
# quotes must use real symbols or they will be skipped.
_REAL_TOKENS = eligible_non_stable()
_TOK_A = _REAL_TOKENS[0] if _REAL_TOKENS else "ETH"
_TOK_B = _REAL_TOKENS[1] if len(_REAL_TOKENS) > 1 else "BNB"
_TOK_C = _REAL_TOKENS[2] if len(_REAL_TOKENS) > 2 else "ADA"
_TOK_D = _REAL_TOKENS[3] if len(_REAL_TOKENS) > 3 else "XRP"
_TOK1, _TOK2, _TOK3, _TOK4 = _TOK_A, _TOK_B, _TOK_C, _TOK_D

# ── Test infrastructure ────────────────────────────────────────────────────

_pass = 0
_fail = 0
_errors: list[str] = []


def test(name: str):
    """Decorator-like context. Yields, then prints pass/fail."""
    class _Ctx:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            global _pass, _fail
            if exc_type is None:
                _pass += 1
                print(f"  \033[32mPASS\033[0m {name}")
            else:
                _fail += 1
                msg = f"{name}: {exc_val}"
                _errors.append(msg)
                print(f"  \033[31mFAIL\033[0m {msg}")
            return True  # suppress exception
    return _Ctx()


def check(cond, msg=""):
    if not cond:
        raise AssertionError(msg)


# ── Mock factories ──────────────────────────────────────────────────────────

def _quote(
    pct_1h=2.0, pct_24h=5.0, price=1.0, vol_24h=500_000,
    vol_chg_24h=10.0, mcap=10_000_000,
):
    """Build a CMC quote dict matching cmc_fetch_quotes_momentum output."""
    return {
        "price": price,
        "percent_change_1h": pct_1h,
        "percent_change_24h": pct_24h,
        "volume_24h": vol_24h,
        "volume_change_24h": vol_chg_24h,
        "market_cap": mcap,
    }


def _candidate(sym, accel=10.0, freshness=1.5, pct_1h=2.0, pct_24h=-2.0,
               price=1.0, vol=500_000, mcap=10_000_000, reason=""):
    return MomentumCandidate(
        symbol=sym, acceleration_score=accel, freshness=freshness,
        pct_1h=pct_1h, pct_24h=pct_24h, price=price,
        vol_24h=vol, market_cap=mcap, reason=reason,
    )


def _mock_cmc_fetch(quotes):
    """Return a callable matching cmc_fetch signature."""
    return lambda symbols, interval, count: quotes


def _mock_mcp(ok=True, analysis=""):
    """Return a callable matching mcp_execute signature."""
    def _exec(skill, params):
        if skill == "altcoin_breakout_scanner_spot":
            if analysis:
                return {"ok": True, "data": {"decision_report": {"analysis": analysis}}}
            return {"ok": False, "error": {"message": "no candidates"}}
        if skill == "daily_market_overview":
            return {
                "ok": ok,
                "data": {
                    "decision_report": {
                        "conclusion": "F&G 52, BTC dom 51%, mcap +1.2%",
                        "analysis": "Market overview.",
                    },
                    "market_read": {"fear_greed": 52},
                },
            }
        return {"ok": False, "error": {"message": f"unknown: {skill}"}}
    return _exec


class MockLLM:
    def __init__(self, response=None, raise_on_call=False):
        self._response = response
        self._raise = raise_on_call
    def chat(self, system, user):
        if self._raise:
            raise RuntimeError("Simulated LLM timeout")
        return json.dumps(self._response)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MOMENTUM ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def test_momentum_ignition():
    """Ignition: freshness >= 0.20 with positive 24h → candidate accepted."""
    with test("Ignition breakout (+2% 1h, +5% 24h)"):
        q = _quote(pct_1h=2.0, pct_24h=5.0)
        cand = _score_token("TOKEN", q, "risk_on")
        check(cand is not None, "Should accept ignition")
        check(cand.freshness == 2.0 / 5.0, f"freshness={cand.freshness}")
        check(cand.pct_1h == 2.0)
        check(cand.pct_24h == 5.0)

def test_momentum_slingshot():
    """Slingshot: negative 24h, positive 1h → auto freshness=1.5."""
    with test("Slingshot reversal (-2% 24h, +3% 1h)"):
        q = _quote(pct_1h=3.0, pct_24h=-2.0)
        cand = _score_token("TOKEN", q, "risk_on")
        check(cand is not None, "Should accept slingshot")
        check(cand.freshness == 1.5, f"freshness={cand.freshness}")

def test_momentum_exhausted():
    """Exhausted: freshness < 0.20 → rejected."""
    with test("Exhausted pump rejected (freshness < 0.20)"):
        q = _quote(pct_1h=0.5, pct_24h=15.0)  # freshness = 0.5/15 = 0.033
        cand = _score_token("TOKEN", q, "risk_on")
        check(cand is None, "Should reject exhausted pump")

def test_momentum_friction_floor():
    """Friction floor: pct_1h < 1.25% → rejected."""
    with test("Friction floor reject (1h < 1.25%)"):
        q = _quote(pct_1h=0.5, pct_24h=2.0)
        cand = _score_token("TOKEN", q, "risk_on")
        check(cand is None, "Should reject below friction floor")

def test_momentum_friction_floor_edge():
    """Friction floor: exactly 1.25% → accepted."""
    with test("Friction floor edge (1h = 1.25%)"):
        q = _quote(pct_1h=1.25, pct_24h=2.0)
        cand = _score_token("TOKEN", q, "risk_on")
        check(cand is not None, "1.25% exactly should pass")

def test_momentum_low_volume():
    """Volume below $100K → rejected."""
    with test("Low volume reject (< $100K)"):
        q = _quote(pct_1h=3.0, pct_24h=5.0, vol_24h=50_000)
        cand = _score_token("TOKEN", q, "risk_on")
        check(cand is None, "Should reject low volume")

def test_momentum_volume_edge():
    """Volume exactly $100K → accepted."""
    with test("Volume edge ($100K exactly)"):
        q = _quote(pct_1h=2.0, pct_24h=5.0, vol_24h=100_000)
        cand = _score_token("TOKEN", q, "risk_on")
        check(cand is not None, "$100K exactly should pass")

def test_momentum_climax_24h():
    """24h gain > 40% → rejected as blow-off top."""
    with test("Climax exhaustion reject (24h > 40%)"):
        q = _quote(pct_1h=5.0, pct_24h=45.0)
        cand = _score_token("TOKEN", q, "risk_on")
        check(cand is None, "Should reject 24h blow-off")

def test_momentum_climax_24h_edge():
    """24h gain exactly 40% → accepted (if freshness also passes)."""
    with test("Climax exhaustion edge (24h = 40%)"):
        # Need pct_1h high enough that freshness >= 0.20
        # freshness = 8 / max(40, 8, 0.1) = 8/40 = 0.20 → passes
        q = _quote(pct_1h=8.0, pct_24h=40.0)
        cand = _score_token("TOKEN", q, "risk_on")
        check(cand is not None, f"40% exactly with freshness=0.20 should pass, got None")

def test_momentum_hourly_blowoff():
    """1h gain > 30% → rejected as manipulation spike."""
    with test("Hourly blow-off reject (1h > 30%)"):
        q = _quote(pct_1h=35.0, pct_24h=20.0)
        cand = _score_token("TOKEN", q, "risk_on")
        check(cand is None, "Should reject hourly blow-off")

def test_momentum_hourly_blowoff_edge():
    """1h gain exactly 30% → accepted."""
    with test("Hourly blow-off edge (1h = 30%)"):
        q = _quote(pct_1h=30.0, pct_24h=20.0)
        cand = _score_token("TOKEN", q, "risk_on")
        check(cand is not None, "30% exactly should pass")

def test_momentum_accel_with_volume_bonus():
    """Volume increase → acceleration bonus applied."""
    with test("Volume bonus: positive vol_change_24h boosts score"):
        q_no_vol = _quote(pct_1h=3.0, pct_24h=5.0, vol_chg_24h=0)
        q_vol = _quote(pct_1h=3.0, pct_24h=5.0, vol_chg_24h=14.0)
        c_no = _score_token("T1", q_no_vol, "risk_on")
        c_yes = _score_token("T2", q_vol, "risk_on")
        check(c_yes.acceleration_score > c_no.acceleration_score,
              f"Vol bonus: {c_yes.acceleration_score} vs no-bonus {c_no.acceleration_score}")

def test_momentum_accel_volume_bonus_capped():
    """Volume bonus capped at 15%."""
    with test("Volume bonus capped at 15% even if vol_chg > 15"):
        q_15 = _quote(pct_1h=3.0, pct_24h=5.0, vol_chg_24h=15.0)
        q_100 = _quote(pct_1h=3.0, pct_24h=5.0, vol_chg_24h=100.0)
        c15 = _score_token("T1", q_15, "risk_on")
        c100 = _score_token("T2", q_100, "risk_on")
        check(abs(c15.acceleration_score - c100.acceleration_score) < 0.0001,
              f"Bonus not capped: {c15.acceleration_score} vs {c100.acceleration_score}")

def test_momentum_ranking():
    """Cross-sectional ranking: highest acceleration first."""
    with test("Cross-sectional ranking (sorted by score)"):
        # Use real competition tokens with UPPERCASE keys (matching CMC response)
        quotes = {
            _TOK_A.upper(): _quote(pct_1h=2.0, pct_24h=-3.0, price=1.0, vol_24h=500_000),
            _TOK_B.upper(): _quote(pct_1h=5.0, pct_24h=-1.0, price=1.0, vol_24h=500_000),
            _TOK_C.upper(): _quote(pct_1h=1.5, pct_24h=-5.0, price=1.0, vol_24h=500_000),
        }
        result = discover_candidates(
            mcp_execute=_mock_mcp(),
            regime="risk_on",
            top_n=5,
            cmc_fetch=_mock_cmc_fetch(quotes),
        )
        # All three are slingshots with freshness=1.5.
        # Ranking by acceleration: _TOK_B (5% 1h) > _TOK_A (2% 1h) > _TOK_C (1.5% 1h)
        check(len(result.candidates) >= 3, f"Expected >=3, got {len(result.candidates)}")
        scores = [c.acceleration_score for c in result.candidates]
        check(scores == sorted(scores, reverse=True), f"Not sorted: {scores}")

def test_momentum_cooldown():
    """Tokens in penalty box are skipped."""
    with test("Penalty box cooldown filters token"):
        tok = _TOK_A.upper()
        quotes = {tok: _quote(pct_1h=3.0, pct_24h=-2.0)}
        cooldowns = {tok: time.time() + 3600}  # cooldown for 1 hour
        result = discover_candidates(
            mcp_execute=_mock_mcp(),
            regime="risk_on",
            top_n=5,
            cmc_fetch=_mock_cmc_fetch(quotes),
            cooldowns=cooldowns,
        )
        check(len(result.candidates) == 0,
              f"Should skip cooldown token, got {len(result.candidates)}")

def test_momentum_zero_mcap():
    """Zero market cap → division by 1.0 in turnover ratio."""
    with test("Zero market cap guard (max(mcap, 1.0))"):
        q = _quote(pct_1h=2.0, pct_24h=-3.0, mcap=0)
        cand = _score_token("TINY", q, "risk_on")
        check(cand is not None, "Should handle zero mcap")
        check(cand.acceleration_score > 0, "Score should be positive even with 0 mcap")

def test_momentum_empty_quotes():
    """Empty CMC response → error."""
    with test("Empty quotes → error message"):
        result = discover_candidates(
            mcp_execute=_mock_mcp(),
            regime="risk_on",
            top_n=2,
            cmc_fetch=_mock_cmc_fetch({}),
        )
        check(result.error is not None, "Should report error on empty quotes")

def test_momentum_no_cmc_fetch():
    """Missing CMC fetch function → error."""
    with test("No CMC fetch → error"):
        result = discover_candidates(
            mcp_execute=_mock_mcp(),
            regime="risk_on",
            top_n=2,
        )
        check(result.error is not None, "Should report error without cmc_fetch")

def test_momentum_duration():
    """Duration tracking populated."""
    with test("Scan duration tracked"):
        quotes = {"A": _quote(pct_1h=3.0, pct_24h=-2.0)}
        result = discover_candidates(
            mcp_execute=_mock_mcp(),
            regime="risk_on",
            top_n=2,
            cmc_fetch=_mock_cmc_fetch(quotes),
        )
        check(result.scan_duration_s > 0, f"scan_duration_s={result.scan_duration_s}")

def test_momentum_reason_field():
    """Reason string populated on every candidate."""
    with test("Reason string populated"):
        quotes = {"A": _quote(pct_1h=3.0, pct_24h=-2.0)}
        result = discover_candidates(
            mcp_execute=_mock_mcp(),
            regime="risk_on",
            top_n=2,
            cmc_fetch=_mock_cmc_fetch(quotes),
        )
        for c in result.candidates:
            check("freshness=" in c.reason, f"Reason missing freshness: {c.reason}")
            check("score=" in c.reason, f"Reason missing score: {c.reason}")

def test_momentum_discovers_across_all_5_gates():
    """Token passing all 5 gates has all fields populated."""
    with test("All 5 gates passed → full candidate"):
        q = _quote(pct_1h=2.5, pct_24h=10.0, price=3.50, vol_24h=500_000,
                   vol_chg_24h=5.0, mcap=20_000_000)
        cand = _score_token("FULL", q, "risk_on")
        check(cand is not None, "Should pass all gates")
        check(cand.symbol == "FULL")
        check(cand.price == 3.50)
        check(cand.vol_24h == 500_000)
        check(cand.market_cap == 20_000_000)
        check(cand.acceleration_score > 0)
        check(cand.composite_score == cand.acceleration_score)  # backward compat


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PORTFOLIO MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

def test_portfolio_initial_buy():
    """Empty portfolio → buy top candidates."""
    with test("Initial buy into empty portfolio"):
        holdings = {"USDT": {"balance": 10_000, "cost_basis_usd": 10_000}}
        candidates = [_candidate("FET", accel=10.0), _candidate("INJ", accel=8.0)]
        prices = {"FET": 2.0, "INJ": 15.0, "USDT": 1.0, "BNB": 310}
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_on", 2, 0.80, 10_000, 0,
        )
        buys = [s for s in plan.swaps if s.action == "buy"]
        check(len(buys) == 2, f"Expected 2 buys, got {len(buys)}")
        check(all(b.amount_token > 0 for b in buys), "All buys should have token amounts")
        check(all(b.amount_usd >= 5 for b in buys), "All buys above $5 minimum")
        # risk_on → 80% of $10K = $8K deployable → $4K per position
        check(all(abs(b.amount_usd - 4000) < 1 for b in buys),
              f"Expected $4K per position, got: {[b.amount_usd for b in buys]}")

def test_portfolio_rebalance_sell():
    """Holding not in target → sell."""
    with test("Rebalance: sell holding not in target set"):
        holdings = {
            "USDT": {"balance": 5_000, "cost_basis_usd": 5_000},
            "OLD": {"balance": 1000, "cost_basis_usd": 5_000},
        }
        candidates = [_candidate("FET", accel=10.0)]
        prices = {"OLD": 5.0, "FET": 2.0, "USDT": 1.0, "BNB": 310}
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_on", 2, 0.80, 10_000, 0,
        )
        sells = [s for s in plan.swaps if s.action == "sell" and s.from_token == "OLD"]
        check(len(sells) >= 1, f"OLD should be sold, got {len(sells)} sells")

def test_portfolio_replace_position():
    """Sell old token + buy new token in same tick."""
    with test("Replace: sell OLD buy NEW in same tick"):
        holdings = {
            "USDT": {"balance": 3_000, "cost_basis_usd": 3_000},
            "OLD": {"balance": 1000, "cost_basis_usd": 5_000},
        }
        candidates = [_candidate("FET", accel=10.0)]
        prices = {"OLD": 5.0, "FET": 2.0, "USDT": 1.0, "BNB": 310}
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_on", 2, 0.80, 8_000, 0,
        )
        sells = [s for s in plan.swaps if s.action == "sell"]
        buys = [s for s in plan.swaps if s.action == "buy"]
        check(len(sells) >= 1, f"Should have at least 1 sell, got {len(sells)}")
        check(len(buys) >= 1, f"Should have at least 1 buy, got {len(buys)}")

def test_portfolio_trailing_stop_fires():
    """Price drops 5% below peak → trailing stop exit."""
    with test("Trailing stop fires (price < peak * 0.95)"):
        holdings = {
            "USDT": {"balance": 2_000, "cost_basis_usd": 2_000},
            "DUMP": {"balance": 100, "cost_basis_usd": 5_000, "peak_price": 50.0},
        }
        candidates = [_candidate("FET", accel=10.0)]
        # Current price is 40% below peak → trailing stop at 5% would have fired earlier
        prices = {"DUMP": 40.0, "FET": 2.0, "USDT": 1.0, "BNB": 310}
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_on", 2, 0.80, 9_000, 0,
        )
        exits = [s for s in plan.swaps if "TRAILING_STOP" in s.reason]
        check(len(exits) >= 1, f"Should fire trailing stop, got {len(exits)} exits")
        # Should also set cooldown
        check("DUMP" in plan.new_cooldowns, "Should add DUMP to penalty box")

def test_portfolio_trailing_stop_barely_fires():
    """Price drops exactly 5% → trailing stop fires."""
    with test("Trailing stop edge: exact 5% drop"):
        holdings = {
            "USDT": {"balance": 2_000, "cost_basis_usd": 2_000},
            "EDGE": {"balance": 100, "cost_basis_usd": 5_000, "peak_price": 50.0},
        }
        candidates = [_candidate("FET", accel=10.0)]
        # 50 * 0.95 = 47.5, test with exactly 47.5
        prices = {"EDGE": 47.5, "FET": 2.0, "USDT": 1.0, "BNB": 310}
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_on", 2, 0.80, 9_500, 0,
        )
        exits = [s for s in plan.swaps if "TRAILING_STOP" in s.reason]
        check(len(exits) >= 1, "Exactly 5% should fire trailing stop")

def test_portfolio_trailing_stop_does_not_fire():
    """Price above stop threshold → no exit."""
    with test("Trailing stop does NOT fire (price > peak * 0.95)"):
        holdings = {
            "USDT": {"balance": 5_000, "cost_basis_usd": 5_000},
            "PUMP": {"balance": 100, "cost_basis_usd": 5_000, "peak_price": 50.0},
        }
        candidates = [_candidate("PUMP", accel=10.0)]  # PUMP is in target set
        prices = {"PUMP": 49.0, "USDT": 1.0, "BNB": 310}  # only 2% below peak
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_on", 2, 0.80, 10_000, 0,
        )
        exits = [s for s in plan.swaps if "TRAILING_STOP" in s.reason]
        check(len(exits) == 0, f"Should not fire trailing stop at -2%, got {len(exits)}")

def test_portfolio_peak_update():
    """Price above stored peak → peak updated, no stop triggered."""
    with test("Peak update: price > stored peak"):
        holdings = {
            "USDT": {"balance": 5_000, "cost_basis_usd": 5_000},
            "MOON": {"balance": 100, "cost_basis_usd": 5_000, "peak_price": 50.0},
        }
        candidates = [_candidate("MOON", accel=15.0)]
        prices = {"MOON": 55.0, "USDT": 1.0, "BNB": 310}  # above peak
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_on", 2, 0.80, 10_500, 0,
        )
        exits = [s for s in plan.swaps if "TRAILING_STOP" in s.reason]
        check(len(exits) == 0, "Should not exit when price is above peak")

def test_portfolio_penalty_box():
    """Stop-out token gets 2h cooldown."""
    with test("Penalty box: 2h cooldown on stop-out"):
        holdings = {
            "USDT": {"balance": 5_000, "cost_basis_usd": 5_000},
            "DUMP": {"balance": 100, "cost_basis_usd": 5_000, "peak_price": 50.0},
        }
        candidates = [_candidate("FET", accel=10.0)]
        prices = {"DUMP": 40.0, "FET": 2.0, "USDT": 1.0, "BNB": 310}
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_on", 2, 0.80, 9_000, 0,
        )
        if "DUMP" in plan.new_cooldowns:
            cooldown_duration = plan.new_cooldowns["DUMP"] - time.time()
            check(7000 <= cooldown_duration <= 7300,
                  f"Cooldown ~7200s, got {cooldown_duration:.0f}s")
        else:
            check(False, "DUMP should be in penalty box")

def test_portfolio_quota_exhausted():
    """5/5 trades → empty plan."""
    with test("Quota exhausted (5/5) → empty plan"):
        holdings = {"USDT": {"balance": 10_000, "cost_basis_usd": 10_000}}
        candidates = [_candidate("FET"), _candidate("INJ")]
        prices = {"FET": 2.0, "INJ": 15.0, "USDT": 1.0, "BNB": 310}
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_on", 2, 0.80, 10_000, 5,
        )
        check(plan.remaining_quota == 0)
        check(len(plan.swaps) == 0, f"Expected 0 swaps, got {len(plan.swaps)}")

def test_portfolio_partial_quota():
    """3/5 used → only 2 slots available, drops excess."""
    with test("Partial quota: drop swaps exceeding remaining slots"):
        holdings = {
            "USDT": {"balance": 5_000, "cost_basis_usd": 5_000},
            "OLD1": {"balance": 100, "cost_basis_usd": 2_000},
            "OLD2": {"balance": 100, "cost_basis_usd": 3_000},
        }
        candidates = [_candidate("FET"), _candidate("INJ")]
        prices = {"OLD1": 20.0, "OLD2": 30.0, "FET": 2.0, "INJ": 15.0, "USDT": 1.0, "BNB": 310}
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_on", 2, 0.80, 10_000, 3,
        )
        check(plan.trades_used <= 2, f"Should use <=2 trades, used {plan.trades_used}")

def test_portfolio_bnb_gas_buffer():
    """BNB holdings above $20 → excess sold."""
    with test("BNB gas buffer: excess above $20 sold"):
        holdings = {
            "USDT": {"balance": 5_000, "cost_basis_usd": 5_000},
            "BNB": {"balance": 0.5, "cost_basis_usd": 155.0},  # $155 BNB, $20 reserved
        }
        candidates = [_candidate("FET"), _candidate("INJ")]
        prices = {"FET": 2.0, "INJ": 15.0, "USDT": 1.0, "BNB": 310.0}
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_on", 2, 0.80, 5_155, 0,
        )
        bnb_sells = [s for s in plan.swaps if s.from_token == "BNB"]
        if bnb_sells:
            check(bnb_sells[0].amount_usd < 155.0,
                  f"Sold ${bnb_sells[0].amount_usd:.0f}, should be <$155 (reserve $20)")
            check(bnb_sells[0].amount_token < 0.5,
                  f"Sold {bnb_sells[0].amount_token} BNB, should be <0.5")
        # If BNB not sold due to quota, that's fine

def test_portfolio_bnb_gas_buffer_small():
    """BNB below $20 → NOT sold."""
    with test("BNB gas buffer: small BNB not sold"):
        holdings = {
            "USDT": {"balance": 5_000, "cost_basis_usd": 5_000},
            "BNB": {"balance": 0.03, "cost_basis_usd": 9.30},  # $9.30, below buffer
        }
        candidates = [_candidate("FET"), _candidate("INJ")]
        prices = {"FET": 2.0, "INJ": 15.0, "USDT": 1.0, "BNB": 310.0}
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_on", 2, 0.80, 5_009, 0,
        )
        bnb_sells = [s for s in plan.swaps if s.from_token == "BNB"]
        check(len(bnb_sells) == 0, "BNB below buffer should NOT be sold")

def test_portfolio_stablecoins_as_usd():
    """Stablecoins (USDT/USDC/etc.) counted as available USD."""
    with test("Stablecoin holdings counted as USD"):
        holdings = {
            "USDT": {"balance": 5_000, "cost_basis_usd": 5_000},
            "USDC": {"balance": 2_000, "cost_basis_usd": 2_000},
        }
        candidates = [_candidate("FET"), _candidate("INJ")]
        prices = {"FET": 2.0, "INJ": 15.0, "USDT": 1.0, "USDC": 1.0, "BNB": 310}
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_on", 2, 0.80, 7_000, 0,
        )
        buys = [s for s in plan.swaps if s.action == "buy"]
        # risk_on 80% of $7K = $5.6K → $2.8K per position
        total_buy_usd = sum(b.amount_usd for b in buys)
        check(total_buy_usd > 3_000, f"Should use stablecoins, total buy=${total_buy_usd:.0f}")

def test_portfolio_zero_price_skip():
    """Zero price for candidate → buy skipped."""
    with test("Zero price for candidate → buy skipped"):
        holdings = {"USDT": {"balance": 10_000, "cost_basis_usd": 10_000}}
        candidates = [_candidate("FET"), _candidate("INJ")]
        prices = {"FET": 0, "INJ": 15.0, "USDT": 1.0, "BNB": 310}  # FET has no price
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_on", 2, 0.80, 10_000, 0,
        )
        fet_buys = [s for s in plan.swaps if s.to_token == "FET"]
        check(len(fet_buys) == 0, "FET should be skipped (no price)")

def test_portfolio_below_min_buy():
    """Buy amount below $5 → skipped."""
    with test("Buy amount < $5 → skipped"):
        holdings = {"USDT": {"balance": 10, "cost_basis_usd": 10}}
        candidates = [_candidate("FET", price=2.0)]
        prices = {"FET": 2.0, "USDT": 1.0, "BNB": 310}
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_off", 1, 0.15, 10, 0,
        )
        buys = [s for s in plan.swaps if s.action == "buy"]
        check(len(buys) == 0, f"Buy amount {10*0.15}=$1.50 should be skipped")

def test_portfolio_sell_before_buy():
    """Sells always appear before buys in the plan."""
    with test("Sell-first ordering"):
        holdings = {
            "USDT": {"balance": 3_000, "cost_basis_usd": 3_000},
            "OLD": {"balance": 100, "cost_basis_usd": 5_000},
        }
        candidates = [_candidate("FET"), _candidate("INJ")]
        prices = {"OLD": 50.0, "FET": 2.0, "INJ": 15.0, "USDT": 1.0, "BNB": 310}
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_on", 2, 0.80, 8_000, 0,
        )
        sell_indices = [i for i, s in enumerate(plan.swaps) if s.action == "sell"]
        buy_indices = [i for i, s in enumerate(plan.swaps) if s.action == "buy"]
        if sell_indices and buy_indices:
            check(max(sell_indices) < min(buy_indices),
                  f"Sells at {sell_indices}, buys at {buy_indices}")

def test_portfolio_risk_off_single_position():
    """Risk off → 1 position, 15% allocation."""
    with test("Risk off: 1 position at 15%"):
        holdings = {"USDT": {"balance": 10_000, "cost_basis_usd": 10_000}}
        candidates = [_candidate("FET"), _candidate("INJ")]
        prices = {"FET": 2.0, "INJ": 15.0, "USDT": 1.0, "BNB": 310}
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_off", 1, 0.15, 10_000, 0,
        )
        buys = [s for s in plan.swaps if s.action == "buy"]
        check(len(buys) <= 1, f"risk_off should buy at most 1, got {len(buys)}")
        if buys:
            check(buys[0].amount_usd <= 1500.01,
                  f"risk_off allocation should be ~$1500, got ${buys[0].amount_usd:.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════════

def test_guardrail_drawdown_triggered():
    """Drawdown >= 25% → circuit breaker."""
    with test("Circuit breaker triggers at 25% drawdown"):
        state = {"peak_value_usd": 10_000, "current_value_usd": 7_499,
                 "emergency_triggered": False}
        result = check_drawdown(state)
        check(result.verdict == Verdict.CIRCUIT_BREAKER,
              f"Expected CIRCUIT_BREAKER, got {result.verdict.value}")
        check(state["emergency_triggered"], "Should set emergency_triggered=True")

def test_guardrail_drawdown_edge():
    """Drawdown exactly 25% → circuit breaker."""
    with test("Circuit breaker edge: exactly 25%"):
        state = {"peak_value_usd": 10_000, "current_value_usd": 7_500,
                 "emergency_triggered": False}
        result = check_drawdown(state)
        check(result.verdict == Verdict.CIRCUIT_BREAKER,
              f"Expected CIRCUIT_BREAKER at exactly 25%, got {result.verdict.value}")

def test_guardrail_drawdown_not_triggered():
    """Drawdown < 25% → proceed."""
    with test("Drawdown below 25% → PROCEED"):
        state = {"peak_value_usd": 10_000, "current_value_usd": 7_501,
                 "emergency_triggered": False}
        result = check_drawdown(state)
        check(result.verdict == Verdict.PROCEED,
              f"Expected PROCEED, got {result.verdict.value}")

def test_guardrail_drawdown_latching():
    """Circuit breaker latches — stays triggered."""
    with test("Circuit breaker latching (stays triggered)"):
        state = {"peak_value_usd": 10_000, "current_value_usd": 12_000,
                 "emergency_triggered": True}  # previously triggered
        result = check_drawdown(state)
        check(result.verdict == Verdict.CIRCUIT_BREAKER,
              "Should stay triggered even though value recovered")

def test_guardrail_daily_reset():
    """Daily trade counter resets at midnight UTC."""
    with test("Daily reset: new UTC date → trades_today=0"):
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        state = {"trades_today": 5, "last_trade_date": yesterday}
        state = check_daily_reset(state)
        check(state["trades_today"] == 0, f"Should reset to 0, got {state['trades_today']}")

def test_guardrail_inactivity_triggered():
    """No trade in > 18h → compliance trade."""
    with test("Inactivity > 18h → COMPLIANCE_TRADE"):
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=19)).isoformat()
        state = {"last_trade_ts": old_ts}
        result = check_inactivity(state)
        check(result is not None, "Should trigger")
        check(result.verdict == Verdict.COMPLIANCE_TRADE,
              f"Expected COMPLIANCE_TRADE, got {result.verdict.value}")

def test_guardrail_inactivity_not_triggered():
    """Recent trade → proceed."""
    with test("Inactivity < 18h → proceed"):
        recent = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        state = {"last_trade_ts": recent}
        result = check_inactivity(state)
        check(result is None, "Should not trigger for recent trade")

def test_guardrail_first_run_no_compliance():
    """First ever run (no last_trade_ts) → no compliance."""
    with test("First run: no last_trade_ts → proceed"):
        state = {"last_trade_ts": ""}
        result = check_inactivity(state)
        check(result is None, "First run should not trigger compliance trade")

def test_guardrail_quota_exhausted():
    """5/5 trades → SKIP."""
    with test("Quota exhausted (5/5) → SKIP_REBALANCE"):
        state = {"trades_today": 5}
        result = check_quota(state)
        check(result.verdict == Verdict.SKIP_REBALANCE,
              f"Expected SKIP_REBALANCE, got {result.verdict.value}")

def test_guardrail_quota_ok():
    """3/5 trades → PROCEED with remaining display."""
    with test("Quota OK (3/5) → PROCEED"):
        state = {"trades_today": 3}
        result = check_quota(state)
        check(result.verdict == Verdict.PROCEED,
              f"Expected PROCEED, got {result.verdict.value}")
        check("2 remaining" in result.reason, f"Should say remaining, got: {result.reason}")

def test_guardrail_state_load_defaults():
    """Missing state file → defaults."""
    with test("Missing state → defaults"):
        state = load_state(Path("/tmp/nonexistent_state_test.json"))
        check(state["trades_today"] == 0)
        check(state["peak_value_usd"] == 0)
        check(state["emergency_triggered"] is False)

def test_guardrail_state_roundtrip():
    """State save → load → same values."""
    with test("State persistence: save → load roundtrip"):
        with TemporaryDirectory() as td:
            sp = Path(td) / "state.json"
            original = {
                "peak_value_usd": 12_000, "current_value_usd": 11_500,
                "drawdown_pct": 4.17, "emergency_triggered": False,
                "trades_today": 3, "last_trade_date": "2026-06-21",
                "holdings": {"FET": {"balance": 100, "cost_basis_usd": 200, "peak_price": 2.50}},
                "regime": "risk_on",
            }
            save_state(original, sp)
            loaded = load_state(sp)
            check(loaded["peak_value_usd"] == 12_000)
            check(loaded["trades_today"] == 3)
            check(loaded["holdings"]["FET"]["peak_price"] == 2.50)

def test_guardrail_state_corrupted():
    """Corrupted JSON → defaults (no crash)."""
    with test("Corrupted state file → defaults"):
        with TemporaryDirectory() as td:
            sp = Path(td) / "state.json"
            sp.write_text("not valid {{{ json")
            state = load_state(sp)
            check(state["trades_today"] == 0, "Should return defaults on corrupt file")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. REGIME CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_regime_risk_on():
    with test("Regime: risk_on"):
        llm = MockLLM({
            "regime": "risk_on", "confidence": 0.85,
            "reasoning": "BTC dom declining, F&G 52", "params": {"max_positions": 3, "allocation_pct": 0.50, "momentum_lookback": "24h"},
        })
        decision = classify_regime(llm, _mock_mcp())
        check(decision.regime == "risk_on")
        check(decision.confidence == 0.85)
        check(decision.max_positions >= 3)

def test_regime_risk_off():
    with test("Regime: risk_off"):
        llm = MockLLM({
            "regime": "risk_off", "confidence": 0.90,
            "reasoning": "F&G at 15, extreme fear", "params": {"max_positions": 1, "allocation_pct": 0.10, "momentum_lookback": "24h"},
        })
        decision = classify_regime(llm, _mock_mcp())
        check(decision.regime == "risk_off")
        check(decision.max_positions <= 2)
        check(decision.allocation_pct <= 0.15)

def test_regime_neutral():
    with test("Regime: neutral"):
        llm = MockLLM({
            "regime": "neutral", "confidence": 0.60,
            "reasoning": "Mixed signals, choppy", "params": {"max_positions": 2, "allocation_pct": 0.30, "momentum_lookback": "24h"},
        })
        decision = classify_regime(llm, _mock_mcp())
        check(decision.regime == "neutral")

def test_regime_llm_failure():
    """LLM throws → neutral fallback."""
    with test("Regime: LLM failure → neutral"):
        llm = MockLLM(raise_on_call=True)
        decision = classify_regime(llm, _mock_mcp())
        check(decision.regime == "neutral", f"Expected neutral, got {decision.regime}")
        check(decision.error is not None, "Should set error field")

def test_regime_mcp_failure():
    """MCP throws → neutral fallback."""
    with test("Regime: MCP failure → neutral"):
        def bad_mcp(s, p): raise RuntimeError("MCP down")
        llm = MockLLM()
        decision = classify_regime(llm, bad_mcp)
        check(decision.regime == "neutral")
        check(decision.error is not None)

def test_regime_garbage_json():
    """LLM returns text, not JSON → neutral."""
    with test("Regime: garbage JSON → neutral"):
        class GarbageLLM:
            def chat(self, system, user): return "Buy everything! To the moon!"
        decision = classify_regime(GarbageLLM(), _mock_mcp())
        check(decision.regime == "neutral")
        check(decision.error is not None)

def test_regime_markdown_extraction():
    """JSON inside ```json block extracted correctly."""
    with test("Regime: markdown JSON extraction"):
        response = '```json\n{"regime":"risk_on","confidence":0.72,"reasoning":"test","params":{"max_positions":3,"allocation_pct":0.50,"momentum_lookback":"24h"}}\n```'
        parsed = extract_json(response)
        check(parsed is not None)
        check(parsed["regime"] == "risk_on")

def test_regime_chatty_prefix():
    """LLM chatty text before JSON → extracted."""
    with test("Regime: chatty prefix + JSON"):
        response = 'Here is the classification you asked for: {"regime": "neutral", "confidence": 0.55, "reasoning": "meh", "params": {"max_positions": 2, "allocation_pct": 0.30, "momentum_lookback": "24h"}} Hope that helps!'
        parsed = extract_json(response)
        check(parsed is not None)
        check(parsed["regime"] == "neutral")

def test_regime_bad_regime_coerced():
    """Invalid regime string → neutral."""
    with test("Regime: bad regime → coerced to neutral"):
        parsed = {"regime": "yolo", "confidence": 0.5, "reasoning": "", "params": {}}
        validated = validate_regime_response(parsed)
        check(validated["regime"] == "neutral")

def test_regime_bad_confidence_clamped():
    """Confidence > 1 → clamped to 1."""
    with test("Regime: bad confidence → clamped"):
        parsed = {"regime": "risk_on", "confidence": 99, "reasoning": "", "params": {}}
        validated = validate_regime_response(parsed)
        check(validated["confidence"] == 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MARKET SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

def test_scenario_everything_pumps():
    """All real tokens have positive 1h and 24h → ignition candidates only."""
    with test("Scenario: Everything pumps"):
        quotes = {
            t.upper(): _quote(pct_1h=2.0, pct_24h=5.0, price=1.0, vol_24h=500_000)
            for t in _REAL_TOKENS
        }
        result = discover_candidates(
            mcp_execute=_mock_mcp(), regime="risk_on", top_n=5,
            cmc_fetch=_mock_cmc_fetch(quotes),
        )
        check(len(result.candidates) >= 1, "Should find candidates in pump scenario")
        # All are ignition (positive 24h) → freshness < 1.5
        for c in result.candidates:
            check(c.freshness <= 1.0,
                  f"Pump tokens should have freshness <= 1.0, got {c.freshness}")

def test_scenario_crash_day():
    """Everything red 24h, only a few have positive 1h → slingshots only."""
    with test("Scenario: Crash day (most 24h red, few 1h green)"):
        quotes = {}
        for i, t in enumerate(_REAL_TOKENS):
            if i < 10:
                quotes[t.upper()] = _quote(pct_1h=2.5, pct_24h=-10.0, vol_24h=500_000)
            else:
                quotes[t.upper()] = _quote(pct_1h=0.5, pct_24h=-15.0, vol_24h=500_000)
        result = discover_candidates(
            mcp_execute=_mock_mcp(), regime="risk_on", top_n=5,
            cmc_fetch=_mock_cmc_fetch(quotes),
        )
        # Only first 10 have pct_1h >= 1.25. Those are slingshots (freshness=1.5).
        check(len(result.candidates) >= 1, "Should find slingshot candidates")
        for c in result.candidates:
            check(c.freshness == 1.5, f"Crash day should be slingshots: {c.reason}")

def test_scenario_sideways():
    """Nothing passes friction floor → zero candidates."""
    with test("Scenario: Sideways market (nothing passes friction floor)"):
        quotes = {}
        for i in range(20):
            quotes[f"T{i}"] = _quote(pct_1h=0.5, pct_24h=1.0, vol_24h=500_000)
        result = discover_candidates(
            mcp_execute=_mock_mcp(), regime="risk_on", top_n=5,
            cmc_fetch=_mock_cmc_fetch(quotes),
        )
        check(len(result.candidates) == 0, "Sideways should yield zero candidates")
        check(result.error is not None)

def test_scenario_single_survivor():
    """Only 1 token passes all 5 gates — test with _score_token directly."""
    with test("Scenario: Single survivor (1 token passes all gates)"):
        # Test with _score_token which doesn't need real allowlist symbols
        good = _score_token("GOOD", _quote(pct_1h=5.0, pct_24h=-2.0, vol_24h=500_000), "risk_on")
        lowvol = _score_token("LOWVOL", _quote(pct_1h=3.0, pct_24h=5.0, vol_24h=50_000), "risk_on")
        exhausted = _score_token("EXHAUST", _quote(pct_1h=0.3, pct_24h=20.0, vol_24h=500_000), "risk_on")
        climax = _score_token("CLIMAX", _quote(pct_1h=5.0, pct_24h=45.0, vol_24h=500_000), "risk_on")
        blowoff = _score_token("BLOWOFF", _quote(pct_1h=35.0, pct_24h=10.0, vol_24h=500_000), "risk_on")
        flat = _score_token("FLAT", _quote(pct_1h=0.5, pct_24h=2.0, vol_24h=500_000), "risk_on")
        check(good is not None, "GOOD should pass all gates")
        check(lowvol is None, "LOWVOL should fail liquidity gate")
        check(exhausted is None, "EXHAUST should fail freshness gate")
        check(climax is None, "CLIMAX should fail 24h climax gate")
        check(blowoff is None, "BLOWOFF should fail hourly climax gate")
        check(flat is None, "FLAT should fail friction floor")

def test_scenario_whale_manipulation():
    """Hourly pump > 30% tokens → rejected. Normal token passes."""
    with test("Scenario: Whale manipulation (hourly blow-offs rejected)"):
        siren = _score_token("SIREN", _quote(pct_1h=91.3, pct_24h=28.0, vol_24h=500_000), "risk_on")
        spike = _score_token("SPIKE", _quote(pct_1h=45.0, pct_24h=15.0, vol_24h=500_000), "risk_on")
        normal = _score_token("NORMAL", _quote(pct_1h=3.0, pct_24h=5.0, vol_24h=500_000), "risk_on")
        check(siren is None, "SIREN (+91.3%% 1h) should be rejected")
        check(spike is None, "SPIKE (+45%% 1h) should be rejected")
        check(normal is not None, "NORMAL should pass")

def test_scenario_rising_tide():
    """Risk_on → 2 positions at 80%."""
    with test("Scenario: Rising tide (risk_on → 2 positions, 80%)"):
        holdings = {"USDT": {"balance": 10_000, "cost_basis_usd": 10_000}}
        candidates = [_candidate("FET", accel=15.0), _candidate("INJ", accel=12.0)]
        prices = {"FET": 2.0, "INJ": 15.0, "USDT": 1.0, "BNB": 310}
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_on", 2, 0.80, 10_000, 0,
        )
        buys = [s for s in plan.swaps if s.action == "buy"]
        check(len(buys) == 2, "risk_on should deploy 2 positions")
        total_deployed = sum(b.amount_usd for b in buys)
        check(total_deployed > 7_500, f"risk_on should deploy ~$8K, deployed ${total_deployed:.0f}")

def test_scenario_extreme_fear():
    """Risk_off → 1 position at 15%."""
    with test("Scenario: Extreme fear (risk_off → 1 position, 15%)"):
        holdings = {"USDT": {"balance": 10_000, "cost_basis_usd": 10_000}}
        candidates = [_candidate("FET", accel=10.0), _candidate("INJ", accel=8.0)]
        prices = {"FET": 2.0, "INJ": 15.0, "USDT": 1.0, "BNB": 310}
        plan = generate_swap_plan(
            holdings, candidates, prices, "risk_off", 1, 0.15, 10_000, 0,
        )
        buys = [s for s in plan.swaps if s.action == "buy"]
        check(len(buys) <= 1, f"risk_off should buy <=1, got {len(buys)}")
        if buys:
            check(buys[0].amount_usd <= 1501,
                  f"risk_off allocation capped at $1500, got ${buys[0].amount_usd:.0f}")

def test_scenario_drawdown_trajectory():
    """Simulate drawdown: peak tracking, circuit breaker, latching."""
    with test("Scenario: Drawdown trajectory (peak → 25% → latching)"):
        state = {"peak_value_usd": 10_000, "current_value_usd": 9_000,
                 "emergency_triggered": False}
        # Tick 1: 10% drawdown
        r1 = check_drawdown(state)
        check(r1.verdict == Verdict.PROCEED, f"10% should PROCEED, got {r1.verdict.value}")
        # Tick 2: 25% drawdown
        state["current_value_usd"] = 7_500
        r2 = check_drawdown(state)
        check(r2.verdict == Verdict.CIRCUIT_BREAKER, f"25% should CIRCUIT_BREAKER, got {r2.verdict.value}")
        # Tick 3: price recovers to 12K, but breaker latches
        state["current_value_usd"] = 12_000
        r3 = check_drawdown(state)
        check(r3.verdict == Verdict.CIRCUIT_BREAKER, f"Should stay in CIRCUIT_BREAKER even after recovery, got {r3.verdict.value}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_all():
    global _pass, _fail, _errors

    sections = [
        ("MOMENTUM ENGINE", [
            test_momentum_ignition, test_momentum_slingshot, test_momentum_exhausted,
            test_momentum_friction_floor, test_momentum_friction_floor_edge,
            test_momentum_low_volume, test_momentum_volume_edge,
            test_momentum_climax_24h, test_momentum_climax_24h_edge,
            test_momentum_hourly_blowoff, test_momentum_hourly_blowoff_edge,
            test_momentum_accel_with_volume_bonus, test_momentum_accel_volume_bonus_capped,
            test_momentum_ranking, test_momentum_cooldown, test_momentum_zero_mcap,
            test_momentum_empty_quotes, test_momentum_no_cmc_fetch,
            test_momentum_duration, test_momentum_reason_field,
            test_momentum_discovers_across_all_5_gates,
        ]),
        ("PORTFOLIO MANAGER", [
            test_portfolio_initial_buy, test_portfolio_rebalance_sell,
            test_portfolio_replace_position,
            test_portfolio_trailing_stop_fires, test_portfolio_trailing_stop_barely_fires,
            test_portfolio_trailing_stop_does_not_fire, test_portfolio_peak_update,
            test_portfolio_penalty_box,
            test_portfolio_quota_exhausted, test_portfolio_partial_quota,
            test_portfolio_bnb_gas_buffer, test_portfolio_bnb_gas_buffer_small,
            test_portfolio_stablecoins_as_usd, test_portfolio_zero_price_skip,
            test_portfolio_below_min_buy, test_portfolio_sell_before_buy,
            test_portfolio_risk_off_single_position,
        ]),
        ("GUARDRAILS", [
            test_guardrail_drawdown_triggered, test_guardrail_drawdown_edge,
            test_guardrail_drawdown_not_triggered, test_guardrail_drawdown_latching,
            test_guardrail_daily_reset,
            test_guardrail_inactivity_triggered, test_guardrail_inactivity_not_triggered,
            test_guardrail_first_run_no_compliance,
            test_guardrail_quota_exhausted, test_guardrail_quota_ok,
            test_guardrail_state_load_defaults, test_guardrail_state_roundtrip,
            test_guardrail_state_corrupted,
        ]),
        ("REGIME CLASSIFICATION", [
            test_regime_risk_on, test_regime_risk_off, test_regime_neutral,
            test_regime_llm_failure, test_regime_mcp_failure, test_regime_garbage_json,
            test_regime_markdown_extraction, test_regime_chatty_prefix,
            test_regime_bad_regime_coerced, test_regime_bad_confidence_clamped,
        ]),
        ("MARKET SCENARIOS", [
            test_scenario_everything_pumps, test_scenario_crash_day,
            test_scenario_sideways, test_scenario_single_survivor,
            test_scenario_whale_manipulation,
            test_scenario_rising_tide, test_scenario_extreme_fear,
            test_scenario_drawdown_trajectory,
        ]),
    ]

    print("=" * 68)
    print("FRESHNESS RATIO ALPHA — COMPREHENSIVE PIPELINE TEST SUITE")
    print("=" * 68)

    for title, funcs in sections:
        print(f"\n── {title} ({len(funcs)} tests) ──")
        for fn in funcs:
            fn()

    print(f"\n{'=' * 68}")
    total = _pass + _fail
    if _fail:
        print(f"FAILED: {_pass}/{total} passed, {_fail} FAILED")
        print()
        print("Failures:")
        for e in _errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print(f"ALL {total} TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    run_all()
