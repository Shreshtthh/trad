"""
CMC API client — wraps CoinMarketCap Pro API endpoints.

Rate limit: 300 calls/min on Basic, 500 on Hobbyist, 1000+ on Pro.
Cache TTLs are tuned to keep us well under that ceiling.

Endpoints consumed:
  /v3/fear-and-greed/latest          hourly
  /v3/fear-and-greed/historical      daily
  /v1/global-metrics/quotes/latest   hourly
  /v1/community/trending/token       hourly
  /v1/content/latest                 hourly
  /v1/k-line/points                  every 15 min
  /v3/index/cmc100-latest            4× daily
"""

import logging
import os
import time
from typing import Any, Optional

import requests

from .cache import cache

log = logging.getLogger(__name__)

BASE_URL = "https://pro-api.coinmarketcap.com"
SESSION = requests.Session()
SESSION.headers["Accept"] = "application/json"
SESSION.headers["Accept-Encoding"] = "gzip"

# ── Rate-limit guard ──
_last_call: float = 0
_MIN_INTERVAL: float = 0.25  # max 4 calls/sec against 300/min ceiling


def _get(path: str, params: Optional[dict] = None, ttl: float = 60) -> dict:
    """GET with cache + rate-limit throttle. Returns parsed JSON body."""
    global _last_call

    key = f"{path}:{sorted(params.items()) if params else ''}"
    cached = cache.get(key)
    if cached is not None:
        return cached

    # Throttle
    elapsed = time.monotonic() - _last_call
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)

    api_key = os.getenv("CMC_API_KEY", "")
    if not api_key:
        raise RuntimeError("CMC_API_KEY env var is required")

    resp = SESSION.get(
        f"{BASE_URL}{path}",
        params=params,
        headers={"X-CMC_PRO_API_KEY": api_key},
        timeout=15,
    )
    _last_call = time.monotonic()

    if resp.status_code == 429:
        log.warning("CMC rate-limited — sleeping 10 s")
        time.sleep(10)
        return _get(path, params, ttl)

    resp.raise_for_status()
    body = resp.json()
    if body.get("status", {}).get("error_code", 0) != 0:
        raise RuntimeError(f"CMC API error: {body['status']}")

    cache.set(key, body, ttl)
    return body


# ────────────────────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────────────────────

def fear_and_greed_latest() -> dict:
    """Return {value, value_classification, timestamp}."""
    body = _get("/v3/fear-and-greed/latest", ttl=900)  # 15 min cache
    return body["data"]


def fear_and_greed_historical(start: int = None, limit: int = 50) -> list[dict]:
    """Return list of {timestamp, value, value_classification}."""
    params = {"limit": limit}
    if start:
        params["start"] = start
    body = _get("/v3/fear-and-greed/historical", params, ttl=3600)
    return body["data"]


def global_metrics_latest() -> dict:
    """Return {total_market_cap, btc_dominance, ...}."""
    body = _get("/v1/global-metrics/quotes/latest", ttl=900)
    return body["data"]


def trending_tokens(limit: int = 10) -> list[dict]:
    """Return trending token list from community activity."""
    body = _get("/v1/community/trending/token", {"limit": limit}, ttl=900)
    return body.get("data", [])


def latest_news(limit: int = 10) -> list[dict]:
    """Return latest news / Alexandria articles."""
    body = _get("/v1/content/latest", {"limit": limit}, ttl=900)
    return body.get("data", [])


def cmc100_latest() -> dict:
    """Return CMC100 index current value."""
    body = _get("/v3/index/cmc100-latest", ttl=3600)
    return body["data"]


def kline_points(
    symbol: str,
    interval: str = "1h",
    count: int = 24,
) -> list[dict]:
    """
    Return OHLCV points for *symbol* (CMC slug or symbol).
    interval: 5m | 15m | 30m | 1h | 4h | 1d
    count: number of candles back from now.
    Returns list of {timestamp, quote: {USD: {open, high, low, close, volume}}}.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "count": count,
    }
    body = _get("/v1/k-line/points", params, ttl=300)  # 5 min cache
    return body.get("data", {}).get("points", [])


def kline_prices_batch(
    symbols: list[str],
    interval: str = "1h",
    count: int = 24,
) -> dict[str, list[dict]]:
    """
    Fetch k-line points for multiple symbols.
    Returns {symbol: [points]}.
    CMC doesn't have a batch endpoint — we call in sequence with throttle.
    For 149 tokens × 1 call each = ~37 seconds at 4 calls/sec.
    """
    results: dict[str, list[dict]] = {}
    for i, sym in enumerate(symbols):
        try:
            results[sym] = kline_points(sym, interval, count)
        except Exception:
            log.debug("kline_points failed for %s (attempt %d)", sym, i + 1)
            results[sym] = []
    return results


def resolve_token_ids(symbols: list[str]) -> dict[str, dict]:
    """
    Resolve CMC slugs/symbols to IDs + metadata.
    Uses cached /v1/cryptocurrency/map if possible.
    """
    body = _get("/v1/cryptocurrency/map", {"symbol": ",".join(symbols[:100])}, ttl=86400)
    mapping: dict[str, dict] = {}
    for item in body.get("data", []):
        mapping[item["symbol"]] = item
    return mapping
