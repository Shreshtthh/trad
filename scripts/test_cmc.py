"""Quick smoke test of the CMC data client. Requires CMC_API_KEY in env."""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "agent", ".env"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from data.cmc_client import (
    fear_and_greed_latest,
    global_metrics_latest,
    trending_tokens,
    kline_points,
    cmc100_latest,
)

if not os.getenv("CMC_API_KEY"):
    print("ERROR: Set CMC_API_KEY in agent/.env first")
    sys.exit(1)

print("=== Fear & Greed ===")
fg = fear_and_greed_latest()
print(f"  Value: {fg['value']} ({fg['value_classification']})")

print("=== Global Metrics ===")
gm = global_metrics_latest()
print(f"  BTC Dominance: {gm.get('btc_dominance', 'N/A')}%")
print(f"  Total MCAP: {gm['quote']['USD'].get('total_market_cap', 'N/A'):,.0f}")

print("=== Trending Tokens ===")
trending = trending_tokens(5)
for t in trending[:5]:
    print(f"  {t.get('name', '?')} ({t.get('symbol', '?')})")

print("=== CMC100 ===")
cmc100 = cmc100_latest()
print(f"  Value: {cmc100.get('value', 'N/A')}")

print("=== K-line (BTC) ===")
kl = kline_points("BTC", "1h", 5)
print(f"  Last 5 candles: {len(kl)} points")
if kl:
    last = kl[-1]
    q = last.get("quote", {}).get("USD", {})
    print(f"  Close: ${q.get('close', 'N/A'):,.2f}")

print("\n✅ All endpoints working")
