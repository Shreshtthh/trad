"""
149 BEP-20 token allowlist for BNB Hack: AI Trading Agent Edition Track 1.

Source: Competition-provided COMP_TOKEN_ALLOWLIST (2026-06-20).
Exact case from the official list is the source of truth — no normalization.

Design:
- COMP_TOKENS is built from the raw source list via _build_registry().
  Solo builder call guarantees exactly one entry per token string.
- Duplicate SLX in the source is handled (single entry preserved).
- is_eligible() does exact case-sensitive lookup.
- sector=None means "unknown" → passes sector filter gate on momentum alone.
"""

from typing import Optional

# ── Raw source list (verbatim from competition — 148 entries, SLX duplicated) ──
_RAW = [
    "ETH", "USDT", "USDC", "XRP", "TRX", "DOGE", "ZEC", "ADA", "LINK", "BCH",
    "DAI", "TON", "USD1", "USDe", "M", "LTC", "AVAX", "SHIB", "XAUt", "WLFI",
    "H", "DOT", "UNI", "ASTER", "DEXE", "USDD", "ETC", "AAVE", "ATOM", "U",
    "STABLE", "FIL", "INJ", "币安人生", "NIGHT", "FET", "TUSD", "BONK", "PENGU", "CAKE",
    "SIREN", "LUNC", "ZRO", "KITE", "FDUSD", "BEAT", "PIEVERSE", "BTT", "NFT", "EDGE",
    "FLOKI", "LDO", "B", "FF", "PENDLE", "NEX", "STG", "AXS", "TWT", "HOME",
    "RAY", "COMP", "GWEI", "XCN", "GENIUS", "XPL", "BAT", "SKYAI", "APE", "IP",
    "SFP", "TAG", "NXPC", "AB", "SAHARA", "1INCH", "CHEEMS", "BANANAS31", "RIVER", "MYX",
    "RAVE", "SNX", "FORM", "LAB", "HTX", "USDf", "CTM", "BDX", "SLX", "UB",
    "DUCKY", "FRAX", "BILL", "WFI", "KOGE", "ALE", "FRXUSD", "USDF", "GOMINING", "VCNT",
    "GUA", "DUSD", "SMILEK", "0G", "BEAM", "MY", "SOON", "REAL", "Q",
    "AIOZ", "ZIG", "YFI", "TAC", "lisUSD", "CYS", "ZAMA", "TRIA", "HUMA", "PLUME",
    "ZIL", "XPR", "ZETA", "BabyDoge", "NILA", "ROSE", "VELO", "UAI", "BRETT", "OPEN",
    "BSB", "TOSHI", "BAS", "ACH", "AXL", "LUR", "ELF", "KAVA", "APR", "IRYS",
    "EURI", "XUSD", "BARD", "DUSK", "SUSHI", "PEAQ", "COAI", "BDCA", "XAUM",
]

# ── Sector classification ──
# Only tokens whose CMC narrative sector is confidently known.
# Everything else → None (pass-through on momentum).
_CONFIRMED_SECTORS: dict[str, str] = {
    # Meme
    "DOGE": "Meme", "SHIB": "Meme", "BONK": "Meme", "FLOKI": "Meme",
    "PENGU": "Meme", "BRETT": "Meme", "CHEEMS": "Meme", "TOSHI": "Meme",
    "BabyDoge": "Meme", "BANANAS31": "Meme", "LUNC": "Meme",
    # AI
    "FET": "AI", "AIOZ": "AI", "UAI": "AI", "COAI": "AI", "SKYAI": "AI",
    # DeFi / DEX
    "AAVE": "DeFi", "SNX": "DeFi", "COMP": "DeFi", "SUSHI": "DeFi",
    "YFI": "DeFi", "UNI": "DeFi", "CAKE": "DeFi", "RAY": "DeFi",
    "1INCH": "DeFi", "LDO": "DeFi", "PENDLE": "DeFi", "DEXE": "DeFi",
    "INJ": "DeFi",
    # Gaming / Metaverse
    "AXS": "Gaming", "APE": "Gaming", "BEAM": "Gaming", "SLX": "Gaming",
    "BTT": "Gaming",
    # Layer 1
    "ETH": "Layer 1", "ADA": "Layer 1", "AVAX": "Layer 1",
    "DOT": "Layer 1", "ATOM": "Layer 1", "ETC": "Layer 1",
    "FIL": "Layer 1", "TRX": "Layer 1", "TON": "Layer 1",
    "KAVA": "Layer 1", "ZIL": "Layer 1", "ZEC": "Layer 1",
    "ROSE": "Layer 1", "ZETA": "Layer 1", "XPR": "Layer 1",
    # L2 / Interop
    "STG": "L2", "ZRO": "L2", "AXL": "L2",
    # Oracle
    "LINK": "Oracle",
    # Payments
    "XRP": "Payments", "LTC": "Payments", "BCH": "Payments",
    # Exchange-based
    "TWT": "Exchange-based",
    # RWA
    "XAUt": "RWA",
    # DePIN
    "GOMINING": "DePIN", "IRYS": "DePIN",
    # Social
    "WLFI": "Social",
    # BRC-20
    "SATS": "BRC-20",
    # Data Availability
    "0G": "Data Availability",
}

# ── USD-pegged stablecoins ──
_STABLES = {
    "USDT", "USDC", "DAI", "TUSD", "FDUSD", "FRAX",
    "USDD", "USD1", "USDe", "USDf", "USDF",
    "DUSD", "XUSD", "FRXUSD", "lisUSD", "STABLE",
}


def _build_registry() -> dict[str, dict]:
    """Build COMP_TOKENS from _RAW + _CONFIRMED_SECTORS + _STABLES.
    One call, one source of truth. Duplicates in _RAW (SLX) are harmless —
    dict assignment is idempotent."""
    registry: dict[str, dict] = {}
    for sym in _RAW:
        is_stable = sym in _STABLES
        sector = _CONFIRMED_SECTORS.get(sym)  # None if unknown
        registry[sym] = {"is_stable": is_stable, "sector": sector}
    return registry


COMP_TOKENS: dict[str, dict] = _build_registry()

# ── Fast lookup sets ──
_ELIGIBLE: set[str] = set(COMP_TOKENS.keys())
_STABLECOINS: set[str] = {s for s, m in COMP_TOKENS.items() if m["is_stable"]}

# ── Public API ──

def is_eligible(symbol: str) -> bool:
    """Exact case-sensitive match against the competition allowlist."""
    return symbol in _ELIGIBLE

def is_stablecoin(symbol: str) -> bool:
    """True if the token is a pegged stablecoin (blocked for stablecoin-only pairs)."""
    return symbol in _STABLECOINS

def get_cmc_sector(symbol: str) -> Optional[str]:
    """Return the CMC sector for a token, or None if unknown.
    None → caller should pass the token through on momentum alone."""
    entry = COMP_TOKENS.get(symbol)
    return entry["sector"] if entry else None

def eligible_symbols() -> list[str]:
    """Return all eligible symbols in sorted order."""
    return sorted(_ELIGIBLE)

def eligible_non_stable() -> list[str]:
    """Return eligible symbols excluding stablecoins."""
    return sorted(_ELIGIBLE - _STABLECOINS)
