"""
Dry-run verification: link Phases 1-4 end-to-end with mocked dependencies.

Run:  cd /root/trad/bnb-hack-agent && python3 verify_pipeline.py

Checks:
  A. Sell-first ordering (sells before buys)
  B. Truncation math (max 5 - trades_today)
  C. Position sizing (equal weight within allocation_pct)
  D. Token amounts (amount_token populated, sells=wallet balance, buys=USD/price)
  E. Slippage multiplier (freed capital haircut by 0.985)
  F. BNB gas buffer (BNB not sold below $20 reserve)
"""

import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))

from strategy.regime import classify_regime
from strategy.momentum import discover_candidates
from strategy.portfolio import generate_swap_plan, SLIPPAGE_MULTIPLIER, BNB_GAS_BUFFER_USD


# ── Mock LLM client ──
class MockLLM:
    def __init__(self, regime_override=None):
        self._regime = regime_override

    def chat(self, system: str, user: str) -> str:
        if self._regime:
            return json.dumps(self._regime)
        return json.dumps({
            "regime": "risk_on",
            "confidence": 0.72,
            "reasoning": "BTC dominance declining, F&G 52 in healthy range, CMC100 trending up",
            "params": {
                "max_positions": 3,
                "allocation_pct": 0.50,
                "momentum_lookback": "24h",
            },
        })


# ── Mock MCP executor ──
def mock_mcp_execute(skill_name: str, params: dict) -> dict:
    if skill_name == "daily_market_overview":
        return {
            "ok": True,
            "data": {
                "decision_report": {
                    "conclusion": "Market trending up, BTC dominance -1.1%, capital rotating to alts.",
                    "analysis": "Fear & Greed at 52, total mcap +3.2% 7d, BTC dom -1.1%.",
                },
                "market_read": {"fear_greed": 52, "btc_dominance": 51.3},
            },
        }
    if skill_name == "altcoin_breakout_scanner_spot":
        return {
            "ok": True,
            "data": {
                "decision_report": {
                    "analysis": (
                        "Market scan complete — ranked breakout candidates.\n\n"
                        "1. **CAKE** — composite score: 0.891. +18.2% 24h price gain, "
                        "+45% 24h volume increase, price ~12.3% above EMA50, RSI at 68, "
                        "expanding MACD, sustainability score is 8.\n\n"
                        "2. **LINK** — composite score: 0.756. +7.2% 24h price gain, "
                        "+18% 24h volume increase, price ~4.3% above EMA50, RSI at 58, "
                        "sustainability score is 6.\n\n"
                        "3. **DOGE** — composite score: 0.712. +12.3% 24h price gain, "
                        "+35% 24h volume increase, price ~2.8% above EMA50, RSI at 64, "
                        "expanding MACD, sustainability score is 5.\n\n"
                    ),
                },
            },
        }
    return {"ok": False, "error": {"message": f"unknown skill: {skill_name}"}}


def mock_cmc_fetch(symbols, interval, count):
    return {}


# ── Mock price cache ──
MOCK_PRICES = {
    "ADA": 0.50,     # $0.50 per ADA
    "CAKE": 2.38,    # $2.38 per CAKE
    "LINK": 15.80,   # $15.80 per LINK
    "DOGE": 0.12,    # $0.12 per DOGE
    "BNB": 310.0,    # $310 per BNB
    "USDT": 1.00,
}


def run_verification():
    errors = []

    print("=" * 60)
    print("PIPELINE DRY-RUN VERIFICATION (v2 — with token amounts, slippage, BNB buffer)")
    print("=" * 60)

    # ─── 1. Phase 3: Regime Classifier ───
    print("\n─── 1. Regime Classifier (Phase 3) ───")

    llm = MockLLM()
    decision = classify_regime(llm, mock_mcp_execute)

    print(f"  Regime: {decision.regime}")
    print(f"  Confidence: {decision.confidence}")
    print(f"  Max positions: {decision.max_positions}")
    print(f"  Allocation: {decision.allocation_pct}")
    print(f"  Error: {decision.error}")

    assert decision.regime == "risk_on", f"Expected risk_on, got {decision.regime}"
    assert decision.error is None, f"Unexpected error: {decision.error}"
    print("  ✅ Regime classified correctly, no errors")

    # ─── 2. Phase 2: Momentum Engine ───
    print("\n─── 2. Momentum Engine (Phase 2) ───")

    momentum_result = discover_candidates(
        mcp_execute=mock_mcp_execute,
        regime=decision.regime,
        hot_sectors=None,
        top_n=5,
        cmc_fetch=mock_cmc_fetch,
    )

    candidates = momentum_result.candidates
    print(f"  Pipeline: {momentum_result.raw_scanned} raw → "
          f"{momentum_result.passed_allowlist} allowlist → "
          f"{momentum_result.passed_stable_gate} non-stable → "
          f"{momentum_result.passed_regime_gate} regime → "
          f"{len(candidates)} final")
    for c in candidates:
        print(f"    {c.symbol}: composite={c.composite_score:.3f}, {c.reason}")

    assert len(candidates) >= 3, f"Expected ≥3 candidates, got {len(candidates)}"
    for i in range(len(candidates) - 1):
        assert candidates[i].composite_score >= candidates[i + 1].composite_score, \
            f"Candidates not sorted"
    print(f"  ✅ Momentum pipeline returned {len(candidates)} sorted candidates")

    # ─── 3. Phase 4: Portfolio Manager ───
    print("\n─── 3. Portfolio Manager (Phase 4) ───")
    print(f"  SLIPPAGE_MULTIPLIER = {SLIPPAGE_MULTIPLIER}")
    print(f"  BNB_GAS_BUFFER_USD   = ${BNB_GAS_BUFFER_USD}")

    # Holdings: USDT $3000, CAKE $1500 (500 tokens), ADA $5000 (10000 tokens), BNB $40
    holdings = {
        "USDT": {"balance": 3000.0, "cost_basis_usd": 3000.0},
        "CAKE": {"balance": 500.0, "cost_basis_usd": 1500.0},
        "ADA":  {"balance": 10000.0, "cost_basis_usd": 5000.0},
        "BNB":  {"balance": 0.129, "cost_basis_usd": 40.0},  # ~$40 worth, $20 reserved
    }
    total_value_usd = 9540.0  # 3000 + 1500 + 5000 + 40
    trades_today = 2

    plan = generate_swap_plan(
        holdings=holdings,
        candidates=candidates,
        price_cache=MOCK_PRICES,
        regime=decision.regime,
        max_positions=decision.max_positions,
        allocation_pct=decision.allocation_pct,
        total_value_usd=total_value_usd,
        trades_today=trades_today,
    )

    print(f"\n  Plan note: {plan.note}")
    print(f"  Trades used: {plan.trades_used}")
    print(f"  Remaining quota: {plan.remaining_quota}")
    print(f"  Idle capital: ${plan.idle_capital_usd:.2f}")
    print(f"\n  SWAP PLAN ({len(plan.swaps)} actions):")
    for i, s in enumerate(plan.swaps):
        tok = f"({s.amount_token:.4f} tokens)" if s.amount_token > 0 else ""
        print(f"    {i+1}. {s.action.upper():4s} {s.from_token:>6s} → {s.to_token:<6s}  "
              f"${s.amount_usd:>8.2f} {tok:>20s} | {s.reason}")

    # ── CHECK A: Sell-first ordering ──
    print("\n  ── Check A: Sell-first ordering ──")
    sell_indices = [i for i, s in enumerate(plan.swaps) if s.action == "sell"]
    buy_indices = [i for i, s in enumerate(plan.swaps) if s.action == "buy"]

    if sell_indices and buy_indices:
        max_sell_idx = max(sell_indices)
        min_buy_idx = min(buy_indices)
        if max_sell_idx < min_buy_idx:
            print(f"  ✅ All SELLs (indices {sell_indices}) precede all BUYs (indices {buy_indices})")
        else:
            msg = f"  ❌ SELL/BUY ordering broken: sell at {max_sell_idx}, buy at {min_buy_idx}"
            print(msg); errors.append(msg)
    elif not sell_indices:
        print("  ⚠️  No sells")
    elif not buy_indices:
        print("  ⚠️  No buys")

    # ── CHECK B: Truncation math ──
    print("\n  ── Check B: Truncation math ──")
    max_allowed = 5 - trades_today  # = 3
    if plan.trades_used <= max_allowed:
        print(f"  ✅ Plan uses {plan.trades_used} trades ≤ {max_allowed} allowed")
    else:
        msg = f"  ❌ Plan uses {plan.trades_used} trades > {max_allowed} allowed!"
        print(msg); errors.append(msg)

    if plan.trades_used + plan.remaining_quota == max_allowed:
        print(f"  ✅ Quota math: {plan.trades_used} + {plan.remaining_quota} = {max_allowed}")
    else:
        msg = f"  ❌ Quota math broken"
        print(msg); errors.append(msg)

    # ── CHECK C: Position sizing ──
    print("\n  ── Check C: Position sizing ──")
    deployable = total_value_usd * decision.allocation_pct
    n_positions = min(decision.max_positions, len(candidates))
    expected_per_position = deployable / max(n_positions, 1)
    print(f"  Deployable: ${deployable:.2f} ({decision.allocation_pct*100:.0f}%)")
    print(f"  Target positions: {n_positions}, per-position cap: ${expected_per_position:.2f}")

    for s in plan.swaps:
        if s.action == "buy":
            if s.amount_usd <= expected_per_position + 0.01:
                print(f"  ✅ BUY {s.to_token}: ${s.amount_usd:.2f} ≤ ${expected_per_position:.2f} cap")
            else:
                msg = f"  ❌ BUY {s.to_token}: ${s.amount_usd:.2f} exceeds ${expected_per_position:.2f} cap!"
                print(msg); errors.append(msg)
            if s.amount_usd >= 5.0:
                print(f"  ✅ BUY {s.to_token}: ${s.amount_usd:.2f} ≥ $5.00 minimum")
            else:
                msg = f"  ❌ BUY {s.to_token}: ${s.amount_usd:.2f} below $5.00 minimum!"
                print(msg); errors.append(msg)

    # ── CHECK D: Token amounts ──
    print("\n  ── Check D: Token amounts populated ──")
    for s in plan.swaps:
        if s.amount_token <= 0:
            msg = f"  ❌ {s.action.upper()} {s.from_token}→{s.to_token}: amount_token=0!"
            print(msg); errors.append(msg)
        elif s.action == "sell":
            expected_bal = holdings.get(s.from_token, {}).get("balance", 0)
            # BNB sells only EXCESS above gas buffer, not full balance
            if s.from_token == "BNB":
                bnb_price = MOCK_PRICES.get("BNB", 310)
                excess_bal = (holdings["BNB"]["cost_basis_usd"] - BNB_GAS_BUFFER_USD) / bnb_price
                expected_bal = max(0, excess_bal)
            if abs(s.amount_token - expected_bal) < 0.0001:
                print(f"  ✅ SELL {s.from_token}: amount_token={s.amount_token:.4f} = expected {expected_bal:.4f}")
            else:
                msg = f"  ❌ SELL {s.from_token}: amount_token={s.amount_token:.4f} ≠ expected {expected_bal:.4f}"
                print(msg); errors.append(msg)
        elif s.action == "buy":
            price = MOCK_PRICES.get(s.to_token, 0)
            expected_tok = s.amount_usd / price if price > 0 else 0
            if abs(s.amount_token - expected_tok) < 0.01:
                print(f"  ✅ BUY {s.to_token}: amount_token={s.amount_token:.4f} = ${s.amount_usd:.2f} / ${price:.2f}")
            else:
                msg = f"  ❌ BUY {s.to_token}: amount_token={s.amount_token:.4f} ≠ ${s.amount_usd:.2f} / ${price:.2f}"
                print(msg); errors.append(msg)

    # ── CHECK E: Slippage ──
    print("\n  ── Check E: Slippage multiplier applied ──")
    raw_sell_total = sum(s.amount_usd for s in plan.swaps if s.action == "sell")
    stable_bal = 3000.0
    # Find what available_usd was: stable + sell_proceeds * SLIPPAGE_MULTIPLIER
    # We can verify by checking that idle_capital is less than stable + raw_sell_total
    naive_available = stable_bal + raw_sell_total
    actual_available = plan.idle_capital_usd + sum(s.amount_usd for s in plan.swaps if s.action == "buy")
    if actual_available < naive_available:
        print(f"  ✅ Slippage applied: naive available=${naive_available:.2f}, actual=${actual_available:.2f}")
    else:
        msg = f"  ❌ Slippage not applied: naive={naive_available:.2f}, actual={actual_available:.2f}"
        print(msg); errors.append(msg)

    expected_after_slippage = stable_bal + raw_sell_total * SLIPPAGE_MULTIPLIER
    if abs(actual_available - expected_after_slippage) < 0.01:
        print(f"  ✅ Available_usd = {stable_bal} + {raw_sell_total}×{SLIPPAGE_MULTIPLIER} = {expected_after_slippage:.2f}")
    else:
        msg = f"  ❌ Available_usd mismatch: got {actual_available:.2f}, expected {expected_after_slippage:.2f}"
        print(msg); errors.append(msg)

    # ── CHECK F: BNB gas buffer ──
    print("\n  ── Check F: BNB gas buffer ──")
    bnb_sells = [s for s in plan.swaps if s.from_token == "BNB" and s.action == "sell"]
    bnb_holding = holdings.get("BNB", {}).get("cost_basis_usd", 0)

    # BNB is $40, buffer is $20. Since BNB is NOT in target set {CAKE, LINK, DOGE},
    # but sell_candidates includes BNB excess above $20 only
    if bnb_holding > BNB_GAS_BUFFER_USD:
        excess = bnb_holding - BNB_GAS_BUFFER_USD
        if bnb_sells:
            bnb_sell = bnb_sells[0]
            if abs(bnb_sell.amount_usd - excess) < 0.01:
                print(f"  ✅ BNB excess sold: ${bnb_sell.amount_usd:.2f} = ${bnb_holding:.2f} - ${BNB_GAS_BUFFER_USD:.2f}")
            else:
                msg = f"  ❌ BNB sell amount wrong: ${bnb_sell.amount_usd:.2f} ≠ ${excess:.2f}"
                print(msg); errors.append(msg)
            # Verify amount_token is the excess (NOT full balance)
            if bnb_sell.amount_token < holdings["BNB"]["balance"]:
                print(f"  ✅ BNB sell tokens ({bnb_sell.amount_token:.4f}) < full balance ({holdings['BNB']['balance']})")
            else:
                msg = f"  ❌ BNB selling FULL balance — gas buffer not respected!"
                print(msg); errors.append(msg)
        else:
            print(f"  ℹ️  BNB excess ${excess:.2f} exists but not sold (may be dropped by quota)")
    else:
        print(f"  ✅ BNB holdings (${bnb_holding:.2f}) ≤ buffer (${BNB_GAS_BUFFER_USD}) — nothing to sell")

    # ── EDGE CASES ──
    print("\n─── 4. Edge Case: Quota Exhaustion ───")
    exhausted_plan = generate_swap_plan(
        holdings=holdings, candidates=candidates, price_cache=MOCK_PRICES,
        regime=decision.regime, max_positions=decision.max_positions,
        allocation_pct=decision.allocation_pct, total_value_usd=total_value_usd,
        trades_today=5,
    )
    if exhausted_plan.remaining_quota == 0 and len(exhausted_plan.swaps) == 0:
        print(f"  ✅ Exhausted → 0 swaps, note: '{exhausted_plan.note}'")
    else:
        msg = f"  ❌ Exhausted plan should have 0 swaps, got {exhausted_plan.trades_used}"
        print(msg); errors.append(msg)

    print("\n─── 5. Edge Case: LLM Failure → Neutral Fallback ───")
    class FailingLLM:
        def chat(self, system, user): raise RuntimeError("Simulated API timeout")
    fail = classify_regime(FailingLLM(), mock_mcp_execute)
    if fail.regime == "neutral" and fail.error is not None:
        print(f"  ✅ LLM failure → neutral (error={fail.error[:40]}...)")
    else:
        msg = "  ❌ LLM failure did not fall back to neutral"; print(msg); errors.append(msg)

    print("\n─── 6. Edge Case: MCP Failure → Neutral Fallback ───")
    def failing_mcp(s, p): raise RuntimeError("MCP connection refused")
    fail2 = classify_regime(llm, failing_mcp)
    if fail2.regime == "neutral" and fail2.error is not None:
        print(f"  ✅ MCP failure → neutral (error={fail2.error[:40]}...)")
    else:
        msg = "  ❌ MCP failure did not fall back to neutral"; print(msg); errors.append(msg)

    print("\n─── 7. Edge Case: Garbage JSON → Neutral Fallback ───")
    class GarbageLLM:
        def chat(self, system, user): return "Buy everything! 🚀"
    fail3 = classify_regime(GarbageLLM(), mock_mcp_execute)
    if fail3.regime == "neutral" and fail3.error is not None:
        print(f"  ✅ Garbage response → neutral (error={fail3.error})")
    else:
        msg = "  ❌ Garbage response did not fall back"; print(msg); errors.append(msg)

    # ── FINAL ──
    print("\n" + "=" * 60)
    if errors:
        print(f"❌ VERIFICATION FAILED — {len(errors)} error(s):")
        for e in errors: print(f"   {e}")
        sys.exit(1)
    else:
        print("✅ ALL CHECKS PASSED — 3 fixes verified (token amounts, slippage, BNB buffer).")
        print("   Ready to proceed to Phase 5 (TWAK execution wrapper).")
        sys.exit(0)


if __name__ == "__main__":
    run_verification()
