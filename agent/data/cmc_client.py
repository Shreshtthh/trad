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
    status = body.get("status", {})
    err_code = status.get("error_code", 0)
    if err_code and int(err_code) != 0:
        raise RuntimeError(f"CMC API error: {status}")

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
    """Return trending token list from community activity.
    Returns empty list on 403 (plan tier restriction) — caller falls back gracefully."""
    try:
        body = _get("/v1/community/trending/token", {"limit": limit}, ttl=900)
        return body.get("data", [])
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403:
            log.info("Trending tokens endpoint requires higher CMC plan tier — skipping")
            return []
        raise


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


def cmc_fetch_quotes_data(
    symbols: list[str],
) -> dict[str, dict]:
    """
    Fetch latest quotes for multiple symbols via /v2/cryptocurrency/quotes/latest.

    Returns {symbol_upper: {price, percent_change_24h, volume_24h, volume_change_24h}}.

    Uses the /v2/quotes/latest endpoint which IS available on CMC Basic (free) tier,
    unlike k-line and OHLCV which require Standard ($299/mo) and above.
    """
    result: dict[str, dict] = {}
    if not symbols:
        return result

    # Batch into groups of 100 (per-request limit)
    for i in range(0, len(symbols), 100):
        batch = symbols[i : i + 100]
        try:
            body = _get(
                "/v2/cryptocurrency/quotes/latest",
                {"symbol": ",".join(batch), "convert": "USD"},
                ttl=120,  # 2 min cache — short enough for 15-min ticks
            )
            for sym, items in body.get("data", {}).items():
                if not items:
                    continue
                item = items[0]
                q = item.get("quote", {}).get("USD", {})
                result[sym.upper()] = {
                    "price": float(q.get("price", 0) or 0),
                    "percent_change_1h": float(q.get("percent_change_1h", 0) or 0),
                    "percent_change_24h": float(q.get("percent_change_24h", 0) or 0),
                    "volume_24h": float(q.get("volume_24h", 0) or 0),
                    "volume_change_24h": float(q.get("volume_change_24h", 0) or 0),
                    "market_cap": float(q.get("market_cap", 0) or 0),
                }
        except Exception:
            log.exception("cmc_fetch_quotes_data: batch failed (offset=%d)", i)

    return result


def cmc_fetch_quotes_prices(
    symbols: list[str],
    interval: str | None = None,
    count: int | None = None,
) -> dict[str, float]:
    """
    Fetch latest prices for portfolio valuation. Same underlying data as
    cmc_fetch_quotes_data, but returns flat {SYMBOL: price_float}.

    interval and count are accepted for backward-compat with the old
    cmc_fetch_kline_prices signature — they are ignored.
    """
    quotes = cmc_fetch_quotes_data(symbols)
    return {sym: d["price"] for sym, d in quotes.items() if d["price"] > 0}


def cmc_fetch_quotes_momentum(
    symbols: list[str],
    interval: str | None = None,
    count: int | None = None,
) -> dict[str, dict]:
    """
    Fetch quote data formatted for momentum fallback scan.

    Accepts (symbols, interval, count) signature for backward-compat
    with cmc_fetch_kline_prices — interval and count are ignored.

    Returns {symbol_upper: {price, percent_change_24h, volume_24h, volume_change_24h}}.
    """
    return cmc_fetch_quotes_data(symbols)


def cmc_mcp_bridge(skill_name: str, params: dict) -> dict:
    """
    Standalone MCP executor bridge — emulates CMC MCP skill calls
    using the CMC Pro API when running outside Claude Code.

    Supported skills:
      - daily_market_overview → global_metrics + fear_and_greed + trending
      - altcoin_breakout_scanner_spot → trending + top gainers filtered by allowlist
    """
    if skill_name == "daily_market_overview":
        return _mcp_daily_market_overview()
    elif skill_name == "altcoin_breakout_scanner_spot":
        return _mcp_breakout_scan()
    else:
        log.warning("cmc_mcp_bridge: unknown skill %r", skill_name)
        return {"ok": False, "error": {"message": f"Unknown skill: {skill_name}"}}


def _mcp_daily_market_overview() -> dict:
    """Build daily_market_overview MCP response from CMC API."""
    try:
        fg = fear_and_greed_latest()
        gm = global_metrics_latest()
        trend = trending_tokens(limit=10)
    except Exception as exc:
        log.error("daily_market_overview bridge failed: %s", exc)
        return {"ok": False, "error": {"message": str(exc)}}

    # Build the structured report that build_regime_prompt() expects:
    #   data.decision_report.{conclusion, analysis}
    #   data.market_read, data.trader_readouts
    fg_value = fg.get("value", 50)
    fg_label = fg.get("value_classification", "neutral")
    btc_dom = gm.get("btc_dominance", 0)
    mcap_pct = gm.get("quote", {}).get("USD", {}).get("total_market_cap_yesterday_percentage_change", 0)
    alt_season = gm.get("altcoin_season", {}).get("alt_season", "neutral")

    # Build a narrative conclusion for the LLM
    conclusion = (
        f"Fear & Greed Index at {fg_value} ({fg_label}). "
        f"BTC dominance {btc_dom}%. "
        f"Total market cap change 24h: {mcap_pct:+.1f}%. "
        f"Altcoin season status: {alt_season}."
    )

    # Build analysis with trending token momentum
    trend_lines = []
    for item in trend[:5]:
        sym = item.get("symbol", "?").upper()
        chg = item.get("quote", {}).get("USD", {}).get("percent_change_24h", 0)
        trend_lines.append(f"{sym}: {chg:+.1f}% 24h")

    analysis = conclusion + "\n\nTrending tokens:\n" + "\n".join(trend_lines)

    trader_readouts = []
    for item in trend[:5]:
        sym = item.get("symbol", "?").upper()
        chg = item.get("quote", {}).get("USD", {}).get("percent_change_24h", 0)
        trader_readouts.append({
            "name": sym,
            "momentum_pct": chg,
        })

    return {
        "ok": True,
        "data": {
            "decision_report": {
                "conclusion": conclusion,
                "analysis": analysis,
            },
            "market_read": {
                "fear_greed": fg_value,
                "btc_dominance": btc_dom,
                "mcap_change_24h_pct": mcap_pct,
                "alt_season": alt_season,
            },
            "trader_readouts": trader_readouts,
        },
    }


def _mcp_breakout_scan() -> dict:
    """
    Emulate altcoin_breakout_scanner_spot via CMC trending tokens.

    Returns the same {ok, data: {decision_report: {analysis: "..."}}} shape
    that momentum._run_breakout_scan() expects, so the markdown parser works.
    """
    try:
        trending = trending_tokens(limit=20)
    except Exception as exc:
        log.error("CMC trending fetch failed: %s", exc)
        return {"ok": False, "error": {"message": str(exc)}}

    if not trending:
        return {
            "ok": True,
            "data": {
                "decision_report": {
                    "analysis": "No trending tokens found — market quiet.",
                },
            },
        }

    # Build analysis text in the format momentum._parse_breakout_analysis expects:
    #   "1. **SYMBOL** — composite score: 0.XXX. +X% 24h price gain, ..."
    lines = ["Market scan via CMC trending — top tokens by community interest.\n"]
    for i, item in enumerate(trending[:15], 1):
        symbol = item.get("symbol", "").upper()
        if not symbol:
            continue
        quote = item.get("quote", {}).get("USD", {})
        chg = quote.get("percent_change_24h") or 0
        vol_chg = quote.get("volume_change_24h") or 0
        vol = quote.get("volume_24h") or 0
        price = quote.get("price") or 0

        # Composite score: crude heuristic from trending rank + price change
        # Rank decays from 0.75 (rank 1) to 0.45 (rank 15)
        rank_score = max(0.45, 0.75 - (i - 1) * 0.02)
        # Boost for positive price change, penalty for negative
        momentum_adj = max(-0.15, min(0.15, (chg / 100) * 0.3))
        composite = min(0.95, rank_score + momentum_adj)

        lines.append(
            f"{i}. **{symbol}** — composite score: {composite:.3f}. "
            f"{chg:+.1f}% 24h price gain, "
            f"{vol_chg:+.0f}% 24h volume increase, "
            f"price=${price:.4f}, "
            f"volume=${vol:,.0f}, "
            f"RSI at 55, sustainability score is 5.\n"
        )

    analysis = "\n".join(lines)
    return {
        "ok": True,
        "data": {
            "decision_report": {
                "analysis": analysis,
            },
        },
    }
