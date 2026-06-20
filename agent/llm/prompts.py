"""
LLM prompt templates for regime classification.

The LLM receives structured CMC market data and returns a JSON decision.
It is never on the hot path — fired once per hour with cached result.
"""

import json

# ── System prompt ──

REGIME_SYSTEM_PROMPT = """\
You are a crypto market regime classifier. You receive CoinMarketCap data
and output a STRICT JSON decision. Your job is risk management, not trade
picking. You only see aggregate market data — you do NOT see individual
token prices or momentum scores.

Classification rules:
- risk_off if: Fear & Greed > 75 (extreme greed = distribution risk) OR
  Fear & Greed < 20 (extreme fear = capitulation), OR BTC dominance rising
  sharply (>3% in 7d), OR negative news cascade (regulatory, hacks, macro
  shock), OR global market cap declining >5% in 7d.
- risk_on if: Fear & Greed 35-65 (healthy range), trending tokens have
  real volume (>$1M daily), BTC dominance stable or declining (rotational),
  global market cap trending up over 7d, CMC100 positive.
- neutral otherwise (mixed signals, choppy market, insufficient conviction).

Output ONLY this JSON — no markdown, no backticks, no conversational text:
{"regime":"risk_on|neutral|risk_off","confidence":0.0,"reasoning":"short sentence","params":{"max_positions":3,"allocation_pct":0.35,"momentum_lookback":"24h"}}

Default params by regime: risk_on → 5 pos, 50% alloc. neutral → 3 pos, 35% alloc.
risk_off → 1 pos, 15% alloc (or fully cash if fear is extreme)."""

# ── User message builder ──

def build_regime_prompt(market_data: dict) -> str:
    """
    Build the user message from daily_market_overview MCP output.
    market_data should contain the structured data from the MCP skill
    (decision_report, trader_readouts, macro_deep_read, etc.)
    """
    # Extract the narrative text from the MCP daily_market_overview response.
    # The response has: data.decision_report.conclusion + analysis,
    # plus macro_deep_read, trader_readouts, watchlist sections.
    report = market_data.get("data", market_data)
    decision = report.get("decision_report", {})

    conclusion = decision.get("conclusion", "N/A")
    analysis = decision.get("analysis", "")

    # Build a compact prompt from the available fields
    parts = [
        "=== CMC Daily Market Overview ===",
        f"Summary: {conclusion}",
        "",
    ]

    # Include the full analysis text (already structured by the MCP skill)
    if analysis:
        # Truncate analysis to ~2000 chars to keep prompt compact
        analysis_short = analysis[:2000]
        if len(analysis) > 2000:
            analysis_short += "\n... (truncated for prompt budget)"
        parts.append(f"Full Analysis:\n{analysis_short}")

    raw = "\n".join(parts)

    # Try to extract structured numeric fields if present
    numeric_context = _extract_numeric_context(report)
    if numeric_context:
        raw += f"\n\nKey Metrics:\n{numeric_context}"

    return raw


def _extract_numeric_context(report: dict) -> str:
    """Extract structured numeric fields from the MCP report for the LLM."""
    lines = []

    # Market read section
    market_read = report.get("market_read", {})
    if isinstance(market_read, dict):
        for k, v in market_read.items():
            if isinstance(v, (int, float, str)):
                lines.append(f"  {k}: {v}")

    # Macro deep read
    macro = report.get("macro_deep_read", {})
    if isinstance(macro, dict):
        for k, v in macro.items():
            if isinstance(v, (int, float, str)):
                lines.append(f"  {k}: {v}")

    # Trader readouts — extract key fields
    readouts = report.get("trader_readouts", [])
    if isinstance(readouts, list):
        for item in readouts[:5]:  # top 5 only
            if isinstance(item, dict):
                name = item.get("name") or item.get("symbol", "?")
                momentum = item.get("momentum") or item.get("momentum_pct", "")
                if momentum:
                    lines.append(f"  {name} momentum: {momentum}")

    return "\n".join(lines) if lines else ""


# ── JSON extraction with markdown stripping ──

def extract_json(response_text: str) -> dict | None:
    """
    Extract and parse a JSON object from LLM response text.
    Handles:
      - Raw JSON: '{"regime": "neutral", ...}'
      - Markdown code block: ```json {...} ```
      - Markdown code block without lang: ``` {...} ```
      - LLM chatty prefix: 'Here is the classification: {...}'
      - Trailing text: '{"regime": "neutral"} Hope this helps!'

    Returns parsed dict or None if extraction fails.
    """
    text = response_text.strip()

    # Strip markdown code blocks
    if text.startswith("```"):
        # Find the first newline (end of ```json or ```)
        newline = text.find("\n")
        if newline != -1:
            text = text[newline + 1:]
        # Find the closing ```
        end = text.rfind("```")
        if end != -1:
            text = text[:end]
        text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object boundaries: { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def validate_regime_response(parsed: dict) -> dict:
    """
    Validate and normalize a parsed regime response.
    Returns a safe dict with defaults for any missing/bad fields.
    Never raises — always returns a valid regime dict.
    """
    regime = parsed.get("regime", "neutral")
    if regime not in ("risk_on", "neutral", "risk_off"):
        regime = "neutral"

    confidence = parsed.get("confidence", 0.5)
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    reasoning = str(parsed.get("reasoning", "LLM classification"))[:200]

    params = parsed.get("params", {})
    if not isinstance(params, dict):
        params = {}

    # Apply regime-appropriate defaults
    max_positions = int(params.get("max_positions", 3))
    allocation_pct = float(params.get("allocation_pct", 0.35))

    if regime == "risk_on":
        max_positions = max(max_positions, 3)
        allocation_pct = max(allocation_pct, 0.35)
    elif regime == "risk_off":
        max_positions = min(max_positions, 2)
        allocation_pct = min(allocation_pct, 0.20)

    return {
        "regime": regime,
        "confidence": confidence,
        "reasoning": reasoning,
        "params": {
            "max_positions": max_positions,
            "allocation_pct": allocation_pct,
            "momentum_lookback": params.get("momentum_lookback", "24h"),
        },
    }
