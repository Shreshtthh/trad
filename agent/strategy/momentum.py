"""
Momentum scoring engine — Freshness Ratio Alpha (Free-Tier Optimized).

Strategy: Cross-sectional ranking using 1h vs 24h momentum to find
"Ignition" breakouts (fresh) and "Slingshot" reversals (V-bottom).
Rejects exhausted pumps where most of the move already happened.

Flow (called every 15 min by main.py):
1. MCP breakout scan (usually empty — trending requires Standard tier)
2. Fallback: fetch quotes for ALL non-stable competition tokens via CMC
3. Gate 1: Friction Floor — must have ≥1.25% 1h momentum to clear fees
4. Gate 2: Freshness Multiplier — slingshot vs ignition vs exhausted
5. Gate 3: Liquidity — minimum $100k 24h volume
6. Gate 4: Climax Exhaustion (24h) — reject tokens already up >40%
7. Gate 5: Hourly Climax — reject tokens up >30% in a single hour
8. Cross-sectional ranking by acceleration score → top 2 candidates
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from data.allowlist import COMP_TOKENS, is_eligible, is_stablecoin, get_cmc_sector

log = logging.getLogger(__name__)

# ── Competition-wide allowlist ──
# Scan ALL non-stable BEP-20 tokens from the 149 competition list.
# Previously used a curated 20-token high-beta subset, but real-time
# market data showed actual movers (SKYAI +3.21%, IP +2.62%) were
# outside that subset. Full-list scan = 2 CMC API calls (100 + 49 tokens),
# well within Basic tier rate limits at 4 calls/sec.
def _build_scan_list() -> set[str]:
    """Build the set of tokens to scan from the competition allowlist."""
    from data.allowlist import eligible_non_stable
    tokens = eligible_non_stable()
    # Remove tokens with known CMC symbol issues or delisted
    # (empty for now, but this is the hook)
    return set(tokens)


SCAN_LIST: set[str] = _build_scan_list()

# ── Gates ──
FRICTION_FLOOR_PCT = 1.25       # Must beat round-trip fees (~0.55%) in 1h alone with margin
MIN_VOLUME_24H = 100_000        # Minimum 24h volume to avoid shallow pools
CLIMAX_EXHAUSTION_PCT = 40.0    # Reject if 24h gain >40% (blow-off top)
HOURLY_CLIMAX_PCT = 30.0         # Reject if 1h gain >30% (intra-hour blow-off)
FRESHNESS_MIN = 0.20            # Below 0.20 = exhausted pump, reject

# ── Data model ──

@dataclass
class MomentumCandidate:
    symbol: str
    acceleration_score: float = 0.0  # cross-sectional rank score
    freshness: float = 0.0           # 1h/24h ratio (or 1.5 for slingshot)
    pct_1h: float = 0.0
    pct_24h: float = 0.0
    vol_24h: float = 0.0
    vol_chg_24h: float = 0.0
    market_cap: float = 0.0
    price: float = 0.0
    sector: Optional[str] = None
    reason: str = ""
    # Backward compat fields
    composite_score: float = 0.0
    price_change_24h_pct: float = 0.0
    volume_change_24h_pct: float = 0.0
    rsi_4h: float = 0.0
    close_vs_ema50_pct: float = 0.0
    macd_expanding: bool = False
    narrative_score: int = 0


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
    top_n: int = 2,
    cmc_fetch=None,
    cooldowns: Optional[dict[str, float]] = None,
) -> MomentumResult:
    """
    Cross-sectional momentum discovery using Freshness Ratio.

    Args:
        mcp_execute: MCP client (unused — kept for compat)
        regime: "risk_on" | "neutral" | "risk_off"
        hot_sectors: unused (freshness ratio replaces sector gate)
        top_n: max candidates (default 2 for concentrated bets)
        cmc_fetch: callable(symbols, interval, count) → dict[str, dict]
        cooldowns: {symbol: cooldown_until_ts} from penalty box
    """
    import time as _time
    t0 = _time.monotonic()
    if cooldowns is None:
        cooldowns = {}

    # Step 1 — Try MCP scan (almost always empty on Basic tier)
    try:
        raw = _run_breakout_scan(mcp_execute)
    except Exception:
        raw = None

    if raw:
        eligible = [c for c in raw if is_eligible(c.symbol)]
        if eligible:
            return _build_result(eligible, top_n, t0, len(raw), len(eligible))

    # Step 2 — Fallback: fetch quotes for high-beta tokens
    if cmc_fetch is None:
        return MomentumResult(error="No CMC fetch function provided")

    try:
        quotes = cmc_fetch(list(SCAN_LIST), None, None)
    except Exception as exc:
        log.error("Quotes fetch failed: %s", exc)
        return MomentumResult(error=f"Quotes fetch failed: {exc}")

    if not quotes:
        return MomentumResult(error="Quotes returned empty")

    result = MomentumResult(raw_scanned=len(SCAN_LIST))

    # Step 3 — Score each token
    scored: list[MomentumCandidate] = []
    for sym in SCAN_LIST:
        # CMC uppercases all response keys, but SCAN_LIST preserves
        # allowlist case (e.g., "BabyDoge", "XAUt"). Always lookup by upper.
        q = quotes.get(sym.upper())
        if not q:
            continue

        # Skip tokens in penalty box
        cooldown_until = cooldowns.get(sym.upper(), 0)
        if _time.time() < cooldown_until:
            log.debug("Penalty box: %s cooldown until %s", sym,
                      _time.strftime("%H:%M:%S", _time.localtime(cooldown_until)))
            continue

        cand = _score_token(sym, q, regime)
        if cand is not None:
            scored.append(cand)

    result.passed_allowlist = len(scored)

    # Step 4 — Cross-sectional ranking
    scored.sort(key=lambda c: c.acceleration_score, reverse=True)
    final = scored[:top_n]

    for c in final:
        result.passed_stable_gate += 1
        result.passed_regime_gate += 1
        c.reason = (
            f"freshness={c.freshness:.2f} "
            f"1h={c.pct_1h:+.1f}% "
            f"24h={c.pct_24h:+.1f}% "
            f"score={c.acceleration_score:.3f}"
        )

    result.candidates = final
    result.scan_duration_s = _time.monotonic() - t0

    log.info(
        "Alpha scan: %d tokens → %d passed gates → %d final (%.1fs)",
        len(SCAN_LIST), len(scored), len(final),
        result.scan_duration_s,
    )
    if not final:
        result.error = "No tokens passed freshness gates"
    return result


# ── Token scoring ──

def _score_token(
    sym: str,
    q: dict,
    regime: str,
) -> Optional[MomentumCandidate]:
    """Score a single token. Returns None if it fails any gate."""

    pct_1h = q.get("percent_change_1h", 0.0)
    pct_24h = q.get("percent_change_24h", 0.0)
    price = q.get("price", 0.0)
    vol_24h = q.get("volume_24h", 0.0)
    vol_chg_24h = q.get("volume_change_24h", 0.0)
    mcap = q.get("market_cap", 1.0)

    # ── Gate 1: Friction Floor ──
    # Must be moving ≥2% per hour to beat round-trip fees.
    if pct_1h < FRICTION_FLOOR_PCT:
        return None

    # ── Gate 2: Freshness Multiplier ──
    if pct_24h < 0:
        # Slingshot: down on the day, violently up this hour. Reversal.
        freshness = 1.5
    else:
        # Ignition: what % of 24h move happened in the last hour?
        freshness = pct_1h / max(pct_24h, pct_1h, 0.1)

    if freshness < FRESHNESS_MIN:
        log.debug("Alpha rejected %s: exhausted pump (freshness=%.2f)", sym, freshness)
        return None

    # ── Gate 3: Liquidity ──
    if vol_24h < MIN_VOLUME_24H:
        log.debug("Alpha rejected %s: low volume ($%.0f)", sym, vol_24h)
        return None

    # ── Gate 4: Climax Exhaustion (24h) ──
    if pct_24h > CLIMAX_EXHAUSTION_PCT:
        log.debug("Alpha rejected %s: climax exhaustion (+%.1f%%)", sym, pct_24h)
        return None

    # ── Gate 5: Hourly Climax ──
    # A token pumping >30% in a single hour is a blow-off, even
    # if the 24h number looks innocent. SIREN +91.3% 1h is the
    # canonical example: massive hourly candle = manipulation.
    if pct_1h > HOURLY_CLIMAX_PCT:
        log.debug("Alpha rejected %s: hourly blow-off (+%.1f%% 1h)", sym, pct_1h)
        return None

    # ── Acceleration Score ──
    # Combines: 1h momentum × freshness × liquidity intensity
    turnover_ratio = vol_24h / max(mcap, 1.0)
    liquidity_intensity = math.log1p(turnover_ratio * 100)
    acceleration = pct_1h * freshness * liquidity_intensity

    # Volume confirmation modifier
    if vol_chg_24h > 0:
        acceleration *= 1.0 + min(vol_chg_24h / 100, 0.15)

    cand = MomentumCandidate(
        symbol=sym.upper(),
        acceleration_score=acceleration,
        freshness=freshness,
        pct_1h=pct_1h,
        pct_24h=pct_24h,
        vol_24h=vol_24h,
        vol_chg_24h=vol_chg_24h,
        market_cap=mcap,
        price=price,
        sector=get_cmc_sector(sym),
        # Backward compat
        composite_score=acceleration,
        price_change_24h_pct=pct_24h,
        volume_change_24h_pct=vol_chg_24h,
    )
    return cand


# ── MCP breakout scan (kept for compat, rarely fires) ──

def _run_breakout_scan(mcp_execute) -> Optional[list[MomentumCandidate]]:
    """Try MCP breakout scan. Returns None on failure (expected on Basic tier)."""
    try:
        response = mcp_execute("altcoin_breakout_scanner_spot", {"preview": True})
    except Exception:
        return None

    if not response.get("ok"):
        return None

    data = response.get("data", {})
    report = data.get("decision_report", {})
    analysis_text = report.get("analysis", "")
    if not analysis_text:
        return None

    return _parse_breakout_analysis(analysis_text)


def _parse_breakout_analysis(text: str) -> list[MomentumCandidate]:
    """Parse CMC MCP breakout analysis text. (Legacy, rarely used.)"""
    import re

    candidates: list[MomentumCandidate] = []
    seen: set[str] = set()

    def _extract_preamble_scores(region: str) -> dict[str, float]:
        scores: dict[str, float] = {}
        for m in re.finditer(r'composite\s+score\s*\(([\d.]+)\)', region, re.IGNORECASE):
            val = float(m.group(1))
            if not (0.3 < val < 1.0):
                continue
            before = region[:m.start()]
            sym_match = re.findall(r'\b([A-Z]{2,10})\b', before)
            if sym_match:
                scores[sym_match[-1]] = val
        for m in re.finditer(r'\b([A-Z]{2,10})\s*\(([\d.]+)\)', region):
            sym, val = m.group(1), float(m.group(2))
            if 0.3 < val < 1.0 and sym not in scores:
                scores[sym] = val
        return scores

    preamble_scores = _extract_preamble_scores(text)
    raw_entries = re.split(r'\n(?=\d+\.\s+\*\*)', text)

    for entry in raw_entries:
        sym_match = re.match(r'\d+\.\s+\*\*(\S+)', entry)
        if not sym_match:
            continue
        symbol = sym_match.group(1).rstrip('*')
        seen.add(symbol)

        boundary = re.search(r'\n(?:#{1,3}\s|Backup Watchlist|\d+\.\s+\*\*)', entry)
        entry_clean = entry[:boundary.start()] if boundary else entry

        cand = MomentumCandidate(symbol=symbol)
        score_match = re.search(
            r'composite\s+(?:score(?:\s+of)?\s*:?\s*)?([\d.]+)',
            entry_clean, re.IGNORECASE)
        if score_match:
            cand.composite_score = float(score_match.group(1).rstrip(').:'))
        elif symbol in preamble_scores:
            cand.composite_score = preamble_scores[symbol]

        pm = re.search(r'([\d.-]+)%\s+24h?\s*(?:hour\s*)?price\s+(?:gain|loss|change|rise|drop|surge)', entry_clean)
        if pm:
            cand.price_change_24h_pct = float(pm.group(1))
        vm = re.search(r'([\d.-]+)%\s+24h?\s*(?:hour\s*)?volume\s+(?:increase|decrease|change|rise|surge)', entry_clean)
        if vm:
            cand.volume_change_24h_pct = float(vm.group(1))
        rm = re.search(r'RSI\s+(?:of|at)\s+([\d.]+)', entry_clean)
        if rm:
            cand.rsi_4h = float(rm.group(1).rstrip('.'))
        cand.macd_expanding = 'expanding macd' in entry_clean.lower()
        nm = re.search(r'[Ss]ustainability\s+score\s+is\s+(\d+)', entry_clean)
        if nm:
            cand.narrative_score = int(nm.group(1))
        cand.sector = get_cmc_sector(symbol)
        candidates.append(cand)

    return candidates


def _build_result(
    candidates: list[MomentumCandidate],
    top_n: int,
    t0: float,
    raw_scanned: int,
    passed_allowlist: int,
) -> MomentumResult:
    """Truncate and wrap candidates into a MomentumResult."""
    candidates.sort(key=lambda c: c.composite_score or c.acceleration_score,
                    reverse=True)
    return MomentumResult(
        candidates=candidates[:top_n],
        raw_scanned=raw_scanned,
        passed_allowlist=passed_allowlist,
        passed_stable_gate=len(candidates[:top_n]),
        passed_regime_gate=len(candidates[:top_n]),
        scan_duration_s=__import__("time").monotonic() - t0,
    )
