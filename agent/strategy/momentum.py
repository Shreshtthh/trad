"""
Momentum scoring engine — MCP orchestrator and eligibility filter.

Flow (called every 15 min by main.py):
1. Broad scan via CMC MCP altcoin_breakout_scanner_spot (2000 tokens → top 20 breakouts)
2. Allowlist gate — drop any token not in COMP_TOKENS
3. Stablecoin gate — drop pegged stables (no momentum to catch)
4. Sector/regime gate — in neutral/risk_off, deprioritize tokens in cold sectors
5. Trend alignment gate — optional multi-timeframe check on top candidates
6. Return filtered top 3–5 candidates to main.py

The heavy math (EMA, MACD, RSI, volume ratio) is computed by the CMC MCP skill.
This module is a filter + gate, not a calculator.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from data.allowlist import COMP_TOKENS, is_eligible, is_stablecoin, get_cmc_sector

log = logging.getLogger(__name__)

# ── Data model ──

@dataclass
class MomentumCandidate:
    symbol: str
    composite_score: float = 0.0
    price_change_24h_pct: float = 0.0
    volume_change_24h_pct: float = 0.0
    rsi_4h: float = 0.0
    close_vs_ema50_pct: float = 0.0
    macd_expanding: bool = False
    narrative_score: int = 0
    sector: Optional[str] = None
    reason: str = ""


@dataclass
class MomentumResult:
    candidates: list[MomentumCandidate] = field(default_factory=list)
    raw_scanned: int = 0
    passed_allowlist: int = 0
    passed_stable_gate: int = 0
    passed_regime_gate: int = 0
    scan_duration_s: float = 0
    error: Optional[str] = None


# ── Main entry point ──

def discover_candidates(
    mcp_execute,
    regime: str,
    hot_sectors: Optional[set[str]] = None,
    top_n: int = 5,
    cmc_fetch=None,  # callable(symbols: list[str], interval: str, count: int) → dict
) -> MomentumResult:
    """
    Run the full momentum discovery pipeline.

    Args:
        mcp_execute: callable(skill_name, params) → dict — injected MCP client.
        regime: "risk_on" | "neutral" | "risk_off" from regime.py.
        hot_sectors: set of sector names currently leading (from compare_sector_strength).
        top_n: max candidates to return.
        cmc_fetch: optional fallback callable for kline data when MCP scan yields nothing.
    """
    # Step 1 — Broad scan
    raw = _run_breakout_scan(mcp_execute)
    if raw is None:
        return MomentumResult(error="Breakout scan failed — platform error")

    result = MomentumResult(raw_scanned=len(raw))

    # Step 2 — Allowlist gate
    eligible = [c for c in raw if is_eligible(c.symbol)]
    result.passed_allowlist = len(eligible)

    # ── ZERO-CANDIDATE FALLBACK ──────────────────────────────────
    # When the broad scan yields no eligible tokens (common — the 149 BSC
    # list is a tiny subset of the 2000 tokens scanned), fall back to
    # direct k-line momentum scoring on liquid allowlist tokens.
    # This prevents the bot from sitting idle and only ever hitting the
    # 20-hour compliance trade.
    if not eligible and cmc_fetch is not None:
        log.info("No MCP breakout matches — activating k-line fallback for liquid tokens")
        fallback_result = _fallback_liquid_scan(cmc_fetch, regime, hot_sectors, top_n)
        if fallback_result.candidates:
            return fallback_result
        # If fallback also empty, return the original empty result so
        # main.py can decide to HOLD rather than panic-sell.
        result.error = "No eligible candidates from MCP scan or k-line fallback"
        return result
    # ────────────────────────────────────────────────────────────

    if not eligible:
        log.warning("No breakout candidates passed the allowlist gate")
        return result

    # Step 3 — Stablecoin gate
    non_stable = [c for c in eligible if not is_stablecoin(c.symbol)]
    result.passed_stable_gate = len(non_stable)
    if not non_stable:
        log.warning("All eligible candidates were stablecoins — no tradeable momentum")
        return result

    # Step 4 — Sector/regime gate
    passed = _apply_regime_gate(non_stable, regime, hot_sectors)
    result.passed_regime_gate = len(passed)
    if not passed:
        log.info("No candidates passed the regime gate (regime=%s)", regime)
        return result

    # Step 5 — Assign reasons, sort, truncate
    for c in passed:
        c.reason = _build_reason(c, regime)

    passed.sort(key=lambda c: c.composite_score, reverse=True)
    result.candidates = passed[:top_n]

    log.info(
        "Momentum pipeline: %d raw → %d allowlist → %d non-stable → %d regime → %d final",
        result.raw_scanned,
        result.passed_allowlist,
        result.passed_stable_gate,
        result.passed_regime_gate,
        len(result.candidates),
    )
    return result


# ── Internal helpers ──

def _run_breakout_scan(mcp_execute) -> Optional[list[MomentumCandidate]]:
    """
    Call altcoin_breakout_scanner_spot via MCP.
    Returns parsed MomentumCandidate list or None on failure.
    """
    try:
        response = mcp_execute("altcoin_breakout_scanner_spot", {"preview": True})
    except Exception as exc:
        log.error("MCP breakout scan exception: %s", exc)
        return None

    if not response.get("ok"):
        err = response.get("error", {}).get("message", "unknown MCP error")
        log.error("MCP breakout scan failed: %s", err)
        return None

    data = response.get("data", {})
    report = data.get("decision_report", {})
    analysis_text = report.get("analysis", "")

    # Parse candidates from the analysis text.
    # The altcoin_breakout_scanner_spot returns ranked candidates in the
    # "analysis" field as markdown with numbered entries.
    candidates = _parse_breakout_analysis(analysis_text)
    return candidates


def _parse_breakout_analysis(text: str) -> list[MomentumCandidate]:
    """
    Parse the English analysis text from altcoin_breakout_scanner_spot.

    Three regions:
      A. Ranking preamble — "composite score (0.696)" / "leader SYMBOL (0.XXXX)"
      B. Numbered detail entries — `1. **SYM …**` with RSI, EMA, price/vol
      C. Backup watchlist — inline "SYM (Name, composite 0.XXXX)"
    """
    import re

    candidates: list[MomentumCandidate] = []
    seen: set[str] = set()

    # ── Helper: extract composite scores from ranking preamble ───────
    # Two patterns:
    #   "SUP has … composite score (0.696)" → nearest preceding CAPS word
    #   "leader BOBA (0.6112)"             → symbol directly before paren
    def _extract_preamble_scores(region: str) -> dict[str, float]:
        scores: dict[str, float] = {}
        # Pattern A: "composite score (N.NNN)" → walk back for symbol
        for m in re.finditer(r'composite\s+score\s*\(([\d.]+)\)', region, re.IGNORECASE):
            val = float(m.group(1))
            if not (0.3 < val < 1.0):
                continue
            # Search backwards for the nearest all-caps word (symbol)
            before = region[:m.start()]
            sym_match = re.findall(r'\b([A-Z]{2,10})\b', before)
            if sym_match:
                scores[sym_match[-1]] = val  # closest
        # Pattern B: "SYMBOL (N.NNN)" adjacent
        for m in re.finditer(r'\b([A-Z]{2,10})\s*\(([\d.]+)\)', region):
            sym, val = m.group(1), float(m.group(2))
            if 0.3 < val < 1.0 and sym not in scores:
                scores[sym] = val
        return scores

    preamble_scores: dict[str, float] = _extract_preamble_scores(text)

    # ── Pass B: Numbered detail entries ──────────────────────────────
    # Split on numbered entries, and truncate each at the next section
    # header or backup list to prevent bleed-through.
    raw_entries = re.split(r'\n(?=\d+\.\s+\*\*)', text)
    for entry in raw_entries:
        sym_match = re.match(r'\d+\.\s+\*\*(\S+)', entry)
        if not sym_match:
            continue
        symbol = sym_match.group(1).rstrip('*')
        seen.add(symbol)

        # Truncate at the next section boundary to avoid bleed
        boundary = re.search(r'\n(?:#{1,3}\s|Backup Watchlist|\d+\.\s+\*\*)', entry)
        entry_clean = entry[:boundary.start()] if boundary else entry

        cand = MomentumCandidate(symbol=symbol)

        # Composite: prefer embedded in THIS entry, fall back to preamble
        score_match = re.search(
            r'composite\s+(?:score(?:\s+of)?\s*:?\s*)?([\d.]+)',
            entry_clean, re.IGNORECASE
        )
        if score_match:
            cand.composite_score = float(score_match.group(1).rstrip(').:'))
        elif symbol in preamble_scores:
            cand.composite_score = preamble_scores[symbol]

        # 24h price change
        pm = re.search(r'([\d.-]+)%\s+24h?\s*(?:hour\s*)?price\s+(?:gain|loss|change|rise|drop|surge)', entry_clean)
        if pm:
            cand.price_change_24h_pct = float(pm.group(1))

        # 24h volume change
        vm = re.search(r'([\d.-]+)%\s+24h?\s*(?:hour\s*)?volume\s+(?:increase|decrease|change|rise|surge)', entry_clean)
        if vm:
            cand.volume_change_24h_pct = float(vm.group(1))

        # EMA50
        em = re.search(r'(?:close|price)\s+~?([\d.-]+)%\s+(?:above|below)\s+(?:its\s+)?EMA', entry_clean)
        if em:
            cand.close_vs_ema50_pct = float(em.group(1))

        # RSI
        rm = re.search(r'RSI\s+(?:of|at)\s+([\d.]+)', entry_clean)
        if rm:
            cand.rsi_4h = float(rm.group(1).rstrip('.'))

        cand.macd_expanding = 'expanding macd' in entry_clean.lower()
        nm = re.search(r'[Ss]ustainability\s+score\s+is\s+(\d+)', entry_clean)
        if nm:
            cand.narrative_score = int(nm.group(1))

        cand.sector = get_cmc_sector(symbol)
        candidates.append(cand)

    # ── Pass C: Backup watchlist ─────────────────────────────────────
    # Isolate the backup section to avoid false positives from preamble.
    backup_start = text.find('### Backup')
    if backup_start == -1:
        backup_start = text.find('Backup Watchlist')
    if backup_start == -1:
        backup_start = text.find('backup')
    backup_region = text[backup_start:] if backup_start != -1 else ''

    # "RARE (SuperRare, composite 0.5962)" or "ACE (Fusionist, 0.5543)"
    # Match: SYM (anything but closing paren, optionally "composite" then a float)
    for m in re.finditer(
        r'\b([A-Z][A-Z0-9]{1,10})\s*\(([^)]*?\b([\d.]+)\s*)\)',
        backup_region,
    ):
        symbol = m.group(1)
        if symbol in seen:
            continue
        # Avoid false matches on narrative text like "RSI (81.95)"
        if symbol in ('RSI', 'MACD', 'EMA', 'The', 'From', 'This', 'All'):
            continue
        raw_val = m.group(3)
        try:
            val = float(raw_val)
        except ValueError:
            continue
        # Backup candidates have composite scores in 0.4–0.7 range
        if not (0.3 < val < 0.8):
            continue
        seen.add(symbol)

        cand = MomentumCandidate(symbol=symbol, composite_score=val)
        cand.sector = get_cmc_sector(symbol)
        candidates.append(cand)

    return candidates


# ── Top liquid tokens for zero-candidate fallback ──
# These are the most-liquid non-stablecoin tokens from the 149 list
# most likely to have reliable CMC Pro API k-line data.
_FALLBACK_LIQUID: list[str] = [
    "ETH", "BNB", "XRP", "DOGE", "ADA", "LINK", "BCH",
    "LTC", "AVAX", "DOT", "UNI", "AAVE", "CAKE",
    "TRX", "TON", "SHIB", "FLOKI", "BONK",
    "FET", "INJ", "AXS", "SNX", "COMP",
]


def _fallback_liquid_scan(
    cmc_fetch,  # callable(symbols, interval, count) → dict[str, list[dict]]
    regime: str,
    hot_sectors: Optional[set[str]],
    top_n: int,
) -> MomentumResult:
    """
    Fallback: fetch 24h k-line data for liquid allowlist tokens and compute
    simple momentum scores. Used when the MCP breakout scanner returns zero
    eligible BSC tokens.
    """
    import time as _time
    t0 = _time.monotonic()

    result = MomentumResult(raw_scanned=len(_FALLBACK_LIQUID))

    try:
        kline_data = cmc_fetch(_FALLBACK_LIQUID, "1d", 2)  # 2 daily candles
    except Exception as exc:
        log.error("Fallback k-line fetch failed: %s", exc)
        result.error = f"Fallback k-line fetch failed: {exc}"
        return result

    candidates: list[MomentumCandidate] = []
    for sym in _FALLBACK_LIQUID:
        points = kline_data.get(sym, [])
        if len(points) < 2:
            continue

        # Simple 24h momentum: (close_now - close_24h_ago) / close_24h_ago
        try:
            prev_close = float(points[0]["quote"]["USD"]["close"])
            curr_close = float(points[1]["quote"]["USD"]["close"])
            if prev_close <= 0:
                continue
            pct_24h = ((curr_close - prev_close) / prev_close) * 100
        except (KeyError, TypeError, ValueError):
            continue

        if pct_24h <= 0:
            continue  # Only positive momentum in fallback mode

        cand = MomentumCandidate(
            symbol=sym,
            composite_score=pct_24h / 100,  # normalize to ~0-1 scale
            price_change_24h_pct=pct_24h,
            sector=get_cmc_sector(sym),
            reason=f"fallback: 24h_momentum={pct_24h:.1f}%",
        )
        candidates.append(cand)

    result.passed_allowlist = len(candidates)
    result.passed_stable_gate = len(candidates)  # _FALLBACK_LIQUID has no stables

    # Apply regime gate
    passed = _apply_regime_gate(candidates, regime, hot_sectors)
    result.passed_regime_gate = len(passed)

    for c in passed:
        c.reason = f"{c.reason}, sector={c.sector or '?'}, regime={regime}"

    passed.sort(key=lambda c: c.composite_score, reverse=True)
    result.candidates = passed[:top_n]
    result.scan_duration_s = _time.monotonic() - t0

    log.info(
        "Fallback scan: %d liquid tokens → %d positive momentum → %d final (%.1fs)",
        len(_FALLBACK_LIQUID), len(candidates), len(result.candidates),
        result.scan_duration_s,
    )
    return result


def _apply_regime_gate(
    candidates: list[MomentumCandidate],
    regime: str,
    hot_sectors: Optional[set[str]],
) -> list[MomentumCandidate]:
    """
    Apply sector/regime filtering.

    risk_on:   accept all non-stablecoin eligible tokens.
    neutral:   accept tokens with sector=None (unknown) OR sector in hot_sectors.
    risk_off:  accept only tokens with sector in hot_sectors AND composite > threshold.
    """
    if hot_sectors is None:
        hot_sectors = set()

    if regime == "risk_on":
        return candidates  # all pass

    if regime == "neutral":
        passed = []
        for c in candidates:
            if c.sector is None:
                passed.append(c)  # unknown sector → pass through on momentum
            elif c.sector in hot_sectors:
                passed.append(c)  # confirmed hot sector
            else:
                log.debug("Regime gate dropped %s (sector=%s, not in hot_sectors=%s)",
                          c.symbol, c.sector, hot_sectors)
        return passed

    if regime == "risk_off":
        # Strict: only hot-sector tokens with strong composite scores
        passed = []
        for c in candidates:
            if c.sector is not None and c.sector in hot_sectors and c.composite_score >= 0.4:
                passed.append(c)
            else:
                log.debug("Regime gate (risk_off) dropped %s (sector=%s, score=%.3f)",
                          c.symbol, c.sector, c.composite_score)
        return passed

    return candidates


def _build_reason(c: MomentumCandidate, regime: str) -> str:
    """Build a human-readable reason string for trade logging."""
    parts = [
        f"composite={c.composite_score:.3f}",
        f"Δ24h={c.price_change_24h_pct:.1f}%",
        f"RSI={c.rsi_4h:.0f}",
    ]
    if c.sector:
        parts.append(f"sector={c.sector}")
    else:
        parts.append("sector=unknown(pass-through)")
    parts.append(f"regime={regime}")
    return ", ".join(parts)
