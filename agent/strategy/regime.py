"""
LLM regime classifier — fires once per hour, never on the hot path.

Input:  CMC daily_market_overview MCP output (structured market data)
Output: RegimeDecision dataclass with regime, confidence, and trading params.

JSON guardrail: strips markdown code blocks, conversational text, and
falls back to "neutral" if parsing fails entirely. This prevents a
malformed LLM response from crashing the orchestrator mid-week.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from llm.prompts import (
    REGIME_SYSTEM_PROMPT,
    build_regime_prompt,
    extract_json,
    validate_regime_response,
)

log = logging.getLogger(__name__)


@dataclass
class RegimeDecision:
    regime: str = "neutral"       # "risk_on" | "neutral" | "risk_off"
    confidence: float = 0.5
    reasoning: str = ""
    max_positions: int = 3
    allocation_pct: float = 0.35
    momentum_lookback: str = "24h"
    updated_ts: str = ""           # ISO 8601 timestamp
    error: Optional[str] = None    # non-None if LLM call failed → neutral default


def classify_regime(
    llm_client,        # object with .chat(system_prompt, user_message) → str
    mcp_execute,        # callable(skill_name, params) → dict
) -> RegimeDecision:
    """
    Fetch market overview from CMC MCP, pass to LLM for regime classification.

    Returns a RegimeDecision. On any failure (MCP down, LLM timeout, bad JSON),
    returns neutral regime with error field set — the bot trades cautiously
    rather than crashing.
    """
    # Step 1 — Fetch daily market overview from CMC MCP
    try:
        response = mcp_execute("daily_market_overview", {"preview": True})
    except Exception as exc:
        log.error("MCP daily_market_overview failed: %s", exc)
        return RegimeDecision(
            regime="neutral",
            confidence=0.3,
            reasoning=f"MCP fetch failed: {exc}",
            error=f"MCP fetch failed: {exc}",
            updated_ts=datetime.now(timezone.utc).isoformat(),
        )

    if not response.get("ok"):
        err_msg = response.get("error", {}).get("message", "unknown MCP error")
        log.error("MCP daily_market_overview returned error: %s", err_msg)
        return RegimeDecision(
            regime="neutral",
            confidence=0.3,
            reasoning=f"MCP error: {err_msg}",
            error=err_msg,
            updated_ts=datetime.now(timezone.utc).isoformat(),
        )

    market_data = response.get("data", response)

    # Step 2 — Build LLM prompt
    user_message = build_regime_prompt(market_data)

    # Step 3 — Call LLM
    try:
        response_text = llm_client.chat(
            system=REGIME_SYSTEM_PROMPT,
            user=user_message,
        )
    except Exception as exc:
        log.error("LLM call failed: %s", exc)
        return RegimeDecision(
            regime="neutral",
            confidence=0.3,
            reasoning=f"LLM call failed: {exc}",
            error=f"LLM call failed: {exc}",
            updated_ts=datetime.now(timezone.utc).isoformat(),
        )

    # Step 4 — Extract and validate JSON (with markdown stripping)
    parsed = extract_json(response_text)
    if parsed is None:
        log.warning(
            "LLM response could not be parsed as JSON — response=%.200s",
            response_text,
        )
        return RegimeDecision(
            regime="neutral",
            confidence=0.3,
            reasoning=f"JSON parse failed from: {response_text[:100]}",
            error="JSON parse failed",
            updated_ts=datetime.now(timezone.utc).isoformat(),
        )

    validated = validate_regime_response(parsed)

    log.info(
        "Regime classified: %s (confidence=%.2f) — %s",
        validated["regime"],
        validated["confidence"],
        validated["reasoning"],
    )

    return RegimeDecision(
        regime=validated["regime"],
        confidence=validated["confidence"],
        reasoning=validated["reasoning"],
        max_positions=validated["params"]["max_positions"],
        allocation_pct=validated["params"]["allocation_pct"],
        momentum_lookback=validated["params"]["momentum_lookback"],
        updated_ts=datetime.now(timezone.utc).isoformat(),
    )
