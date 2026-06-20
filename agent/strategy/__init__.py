"""
Strategy pipeline — momentum discovery, regime classification, and portfolio management.

Flow:
  1. regime.classify_regime() — hourly, LLM-driven (never on hot path)
  2. momentum.discover_candidates() — 15-min, MCP + allowlist + regime gate
  3. portfolio.generate_swap_plan() — compare holdings vs candidates, truncate to quota
"""
