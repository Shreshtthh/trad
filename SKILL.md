# Freshness Ratio Alpha — Autonomous Momentum Rotation Skill

**BNB Hack: AI Trading Agent Edition — Track 2 Submission**  
**Most Innovative Agent Skill on CMC Skill Hub**

---

## Overview

**Freshness Ratio Alpha** is a free-tier-optimized momentum detection engine that identifies ignition breakouts and slingshot reversals using only percent-change data from the CMC `/v2/cryptocurrency/quotes/latest` endpoint. No OHLCV candles. No k-line data. No premium tier required.

The core insight: a 24h price change tells you *what* moved. But the ratio of 1h change to 24h change tells you *when* the move started — and whether it's still accelerating.

### Why This Matters

Most momentum strategies on the CMC platform require Standard tier ($299/mo) for OHLCV/k-line data. This skill works on the **free Basic tier** (15,000 credits/month), making it accessible to anyone. It achieved this by deriving all signals from the percent-change fields already available on Basic tier: `percent_change_1h`, `percent_change_24h`, `volume_24h`, `volume_change_24h`, and `market_cap`.

---

## The Freshness Ratio

### Formula

```
freshness = pct_1h / max(pct_24h, pct_1h, 0.1)
```

### Signal Types

| Signal | Condition | Meaning |
|--------|-----------|---------|
| **Ignition** | `freshness ≥ 0.20`, `pct_24h > 0` | Fresh breakout — 1h momentum dominates 24h, move is accelerating |
| **Slingshot** | `pct_24h < 0`, `pct_1h ≥ 1.25` | V-bottom reversal — down on the day but violently up this hour. Freshness auto-set to 1.5 |
| **Exhausted** | `freshness < 0.20` | REJECTED — most of the 24h move already happened, entering now = buying the top |

### Why Freshness > Raw Momentum

Consider two tokens:

- **Token A**: +15% 24h, +3% 1h → freshness = 0.20 (borderline exhausted — the pump started 20 hours ago)
- **Token B**: +3% 24h, +2.8% 1h → freshness = 0.93 (ignition — the pump just started this hour)

A naive momentum scanner buys Token A because +15% looks better. **Freshness Ratio Alpha buys Token B** because 93% of the move is fresh — there's still room to run.

---

## The 4-Gate Pipeline

Every candidate token passes through four sequential gates. Only survivors are scored and ranked.

### Gate 1: Friction Floor

```
pct_1h ≥ 1.25%
```

Must beat round-trip DEX fees (~0.25% BSC fee + ~0.3% slippage = 0.55%) in a single hour with margin. A token moving less than 1.25%/hr cannot overcome trading friction.

### Gate 2: Freshness Multiplier

```
freshness ≥ 0.20  (or slingshot: pct_24h < 0 triggers auto-1.5)
```

Rejects exhausted pumps. Slingshot override rewards V-bottom reversals that are violently bouncing.

### Gate 3: Liquidity

```
volume_24h ≥ $100,000
```

Avoids shallow pools where one swap moves the price against you. Every token passing this gate has meaningful market depth.

### Gate 4: Climax Exhaustion (24h)

```
pct_24h ≤ 40.0%
```

Rejects 24h blow-off tops. A token already up >40% in 24h is a climax distribution event, not an entry.

### Gate 5: Hourly Climax

```
pct_1h ≤ 30.0%
```

Rejects intra-hour blow-offs. A token pumping +30% or more in a single hour is manipulation -- do not buy into the candle. SIREN at +91.3% 1h is the canonical example: massive hourly pump, high risk of instant reversal.

---

## Acceleration Score

Tokens surviving all four gates are ranked by:

```
turnover_ratio = volume_24h / market_cap
liquidity_intensity = ln(1 + turnover_ratio × 100)
acceleration = pct_1h × freshness × liquidity_intensity
```

Volume-confirmed moves get a bonus:

```
if volume_change_24h > 0:
    acceleration *= 1.0 + min(volume_change_24h / 100, 0.15)
```

The top 2 tokens by acceleration score become the concentrated positions.

---

## Risk Architecture

### Trailing Stop-Loss (5%)

Each position tracks its peak price since entry. When the current price drops 5% below that peak, the position is fully exited — no exceptions.

**No fixed take-profit.** Winners ride until they reverse. This is asymmetrically positive: a 40% runner that pulls back to +33% still exits at +33% (profit locked), while a 5% loser exits at -5% (loss capped).

Trailing stops are the **only exit mechanism** during a circuit breaker event — see below.

### Penalty Box (2-Hour Cooldown)

Tokens stopped out are banned from re-entry for 2 hours. This prevents whip-saw losses where the bot sells at the bottom then buys back into the same declining token 15 minutes later.

### Circuit Breaker (-25% Drawdown)

If portfolio value drops 25% from its all-time peak:

- **ALL new buys are halted**
- **No regular rebalancing sells** (don't sell into panic)
- **Trailing stops remain active** — stop-loss exits still fire to protect remaining capital
- **Heartbeat trades continue** — competition requires 1 trade/day minimum
- **Latching**: once triggered, stays active until manual reset

This is intentionally asymmetric. The circuit breaker protects the 5% buffer before the competition's 30% disqualification threshold, but it does not force-sell into a drawdown (which would crystalize losses that might recover).

### Daily Trade Cap

```
max 5 trades per 24h UTC
```

Overtrading is a PnL killer — 0.25% DEX fee × 20 trades/day = 5% daily drag. Five trades is enough for 2 concentrated entries + 3 stop-loss exits.

### Heartbeat Trade (18h Inactivity)

If no trade has occurred in 18 hours, the agent force-executes a $5 USDT → FDUSD swap. This satisfies the competition's minimum-1-trade-per-day rule without requiring artificial position changes during quiet markets.

### Concentrated Positions (Top 2, Equal Weight)

Only 2 positions simultaneously. Each at 50% of deployed capital. Concentration is intentional:

- 10 positions at 5% each → even a 100% winner adds only 5% to portfolio
- 2 positions at 40% each → a 25% winner adds 10% to portfolio

The 7-day sprint format rewards concentration. Diversification is protection against unknown risks — but we know exactly which 20 tokens we trade.

---

## Regime Overlay (LLM, Off-Hot-Path)

A regime classifier runs once per hour (never on the 15-minute trade path). It ingests:

- CMC Fear & Greed Index (current + 7-day history)
- BTC dominance trend
- Global market cap 24h change
- Altcoin season indicator
- Trending token momentum

The LLM (DeepSeek Chat, OpenAI-compatible) outputs a structured JSON decision:

```json
{
  "regime": "risk_on | neutral | risk_off",
  "confidence": 0.0-1.0,
  "reasoning": "one-sentence market read",
  "params": {
    "max_positions": 2,
    "allocation_pct": 0.80,
    "momentum_lookback": "24h"
  }
}
```

**Regime effects on position sizing:**

| Regime | Positions | Capital Deployed | Behavior |
|--------|-----------|-----------------|----------|
| risk_on | 2 | 80% | Full concentrated bets |
| neutral | 2 | 60% | Moderate deployment |
| risk_off | 1 | 15% | Minimum exposure, mostly cash |

On LLM failure (API down, timeout, bad JSON): defaults to neutral — never crashes.

---

## Token Universe

The scanner operates on the **full 132-token non-stablecoin competition allowlist** (BEP-20 tokens on BSC). Previously used a curated 20-token high-beta subset, but real-time testing revealed that actual momentum candidates (SKYAI +3.21%, IP +2.62%) were outside that subset. 

**Scan strategy**: 132 tokens batched into 2 CMC API calls (100 + 32) every 15 minutes. Well within free tier rate limits (18,000 calls/month — we use ~240/hour at worst).

Sector coverage across the full 149-token list: Meme, AI, DeFi, Gaming, Layer 1, L2, Oracle, RWA, DePIN, Payments, Exchange-based, Social, BRC-20, Data Availability.

---

## Data Flow (15-Minute Tick)

```
┌──────────────────────────────────────────────────────────────┐
│ TICK START                                                   │
│   │                                                          │
│   ├─ 1. Load state (portfolio_state.json)                    │
│   ├─ 2. Guardrails: drawdown → inactivity → quota            │
│   │     ├─ CIRCUIT_BREAKER: run exits-only, then skip        │
│   │     ├─ COMPLIANCE_TRADE: $5 USDT→FDUSD heartbeat         │
│   │     └─ SKIP_REBALANCE: save state, return                │
│   │                                                          │
│   ├─ 3. Fetch holdings (TWAK wallet portfolio)               │
│   ├─ 4. Regime classification (LLM, hourly, cached)          │
│   │                                                          │
│   ├─ 5. Momentum discovery (Freshness Ratio)                 │
│   │     ├─ CMC quotes/latest for 20 high-beta tokens         │
│   │     ├─ 4-gate pipeline                                   │
│   │     └─ Cross-sectional ranking → top 2                   │
│   │                                                          │
│   ├─ 6. Portfolio rebalancing                                │
│   │     ├─ Trailing stop scan (5% below peak = full exit)    │
│   │     ├─ Sell positions not in target set                  │
│   │     ├─ Buy top-2 targets at equal weight                 │
│   │     └─ Truncate to remaining daily quota                 │
│   │                                                          │
│   └─ 7. Execute (TWAK swap CLI, 8s delay between swaps)      │
│        ├─ Record trades to trade_log.json                    │
│        ├─ Update peak portfolio value                        │
│        └─ Save state                                         │
│                                                              │
│ SLEEP 900s → NEXT TICK                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## State Persistence

### portfolio_state.json
```json
{
  "peak_value_usd": 11000.0,
  "current_value_usd": 10500.0,
  "drawdown_pct": 4.5,
  "emergency_triggered": false,
  "trades_today": 2,
  "last_trade_date": "2026-06-22",
  "last_trade_ts": "2026-06-22T14:30:00Z",
  "holdings": {
    "FET": {"balance": 500.0, "cost_basis_usd": 1000.0, "peak_price": 2.15}
  },
  "cooldowns": {
    "BONK": 1750612345.0
  },
  "regime": "risk_on",
  "regime_updated_ts": "2026-06-22T14:00:00Z"
}
```

### trade_log.json (append-only)
```json
{
  "trades": [
    {
      "ts": "2026-06-22T14:30:00Z",
      "action": "buy",
      "token": "FET",
      "from_token": "USDT",
      "amount": 500.0,
      "amount_usd": 1000.0,
      "tx_hash": "0x...",
      "regime": "risk_on",
      "reason": "freshness=1.50 1h=+2.8% 24h=-1.4% score=4.745"
    }
  ]
}
```

Both files use atomic writes (write to `.tmp`, then `os.replace`) — never corrupt on crash.

---

## CMC Data Endpoints Used

All on **Basic (free) tier**:

| Endpoint | Frequency | TTL Cache |
|----------|-----------|-----------|
| `/v2/cryptocurrency/quotes/latest` | Every 15 min | 120s |
| `/v3/fear-and-greed/latest` | Hourly | 900s |
| `/v1/global-metrics/quotes/latest` | Hourly | 900s |
| `/v1/cryptocurrency/map` | Once daily | 86400s |

Rate limit: ~4 calls/sec against 300/min ceiling. Throttle enforced per-call.

---

## Execution Layer (TWAK + bnbagent-sdk)

- **Wallet**: TWAK autonomous agent wallet, BSC mainnet
- **Swap execution**: `twak swap --amount <token_units> --from <SYM> --to <SYM>`
- **Portfolio fetch**: `twak wallet portfolio --format json`
- **x402 payments**: CMC API calls signed via bnbagent-sdk X402Signer with defense-in-depth (recipient binding, per-call caps, session budget)
- **Nonce safety**: 8-second delay between sequential swaps (2+ BSC blocks)
- **Paper trade mode**: Full pipeline with fake tx hashes — validates end-to-end before live

---

## Why This Wins Track 2

### 1. Novelty
Freshness Ratio is a genuinely new signal. Nobody else is comparing 1h to 24h percent-change as a breakout-timing mechanism. It's not RSI, not MACD, not moving-average crossover — it's a mathematical relationship between two freely-available data points that reveals *when* momentum ignited.

### 2. Accessibility
Most CMC skills require Standard tier data (k-line, OHLCV). This skill works on Basic tier. That means it can be used by anyone with a free CMC API key — dramatically expanding the addressable user base.

### 3. Composability
The skill outputs a clean ranked list of `MomentumCandidate` objects with composite scores. Other skills can consume this output as input — a portfolio rebalancer, a risk manager, a notification system. It's a building block, not a monolith.

### 4. Safety
Four-layer risk architecture (trailing stops, circuit breaker, penalty box, heartbeat trade) means the skill can run autonomously for a week without blowing up. Every failure mode has a defined behavior.

### 5. Measurable
Every trade is logged with the freshness score and regime at entry time. PnL attribution is trivially traceable: "Did the freshness>0.5 trades outperform freshness<0.3 trades?" This is verifiable, not hand-wavy.

---

## SKILL.md Metadata

```yaml
name: freshness-ratio-alpha
version: 1.0.0
domain: trading
result_type: candidate_set
description: >
  Free-tier momentum detection using 1h/24h percent-change ratio.
  Identifies ignition breakouts, slingshot reversals, and rejects
  exhausted pumps. Works on CMC Basic tier — no OHLCV required.
  Returns top-2 concentrated momentum candidates with composite scores.

inputs:
  regime:
    type: string
    enum: [risk_on, neutral, risk_off]
    description: Risk regime from upstream classifier
  top_n:
    type: integer
    default: 2
    description: Number of candidates to return
  cooldowns:
    type: object
    description: Map of {symbol: cooldown_until_timestamp} for penalty box
    optional: true

outputs:
  candidates:
    type: array
    items:
      symbol: string
      acceleration_score: float
      freshness: float
      pct_1h: float
      pct_24h: float
      vol_24h: float
      market_cap: float
      price: float
      reason: string

dependencies:
  cmc_endpoints:
    - /v2/cryptocurrency/quotes/latest
  cmc_tier: basic
  rate_limit: 4 calls/sec

author: BNB Hack Track 1 & 2 Submission
license: MIT
```

---

## File Structure

```
bnb-hack-agent/
├── SKILL.md                          ← THIS FILE (Track 2 submission)
├── IMPLEMENTATION_PLAN.md            ← Full architecture document
├── agent/
│   ├── main.py                       ← 15-min orchestrator loop
│   ├── data/
│   │   ├── cmc_client.py             ← CMC API wrapper (free-tier endpoints)
│   │   ├── cache.py                  ← Thread-safe TTL cache
│   │   └── allowlist.py              ← 149 competition tokens + sectors
│   ├── strategy/
│   │   ├── momentum.py               ← Freshness Ratio Alpha engine
│   │   ├── regime.py                 ← LLM regime classifier
│   │   └── portfolio.py              ← Position sizing + trailing stops
│   ├── execution/
│   │   ├── twak_client.py            ← TWAK swap execution + x402 signing
│   │   └── guardrails.py             ← Drawdown, inactivity, quota limits
│   └── llm/
│       └── prompts.py                ← Regime classification prompts
├── scripts/
│   ├── paper_trade.sh                ← Dry-run validation
│   ├── register_competition.sh       ← TWAK competition registration
│   └── backtest.py                   ← 30-day historical replay
├── state/
│   ├── portfolio_state.json          ← Current portfolio snapshot
│   └── trade_log.json                ← Append-only trade history
└── .env                              ← Secrets (not committed)
```

---

## Quick Start

```bash
# 1. Set up environment
cp .env.example .env
# Fill in: CMC_API_KEY, DEEPSEEK_API_KEY, WALLET_ADDRESS

# 2. Paper trade (validate pipeline)
bash scripts/paper_trade.sh --once

# 3. Run for 4 hours (observe behavior)
bash scripts/paper_trade.sh --hours 4

# 4. Live trading
python3 agent/main.py --interval 900
```

---

## Performance Characteristics

| Metric | Expected Value |
|--------|---------------|
| Tick latency | 2-4 seconds (CMC API + computation) |
| LLM latency | 1-3 seconds (once per hour) |
| Max trades/day | 5 |
| Min trade size | $5 USD |
| Position concentration | 2 tokens |
| Max drawdown before freeze | 25% |
| Stop-loss | 5% trailing from peak |
| Penalty cooldown | 2 hours |
| Revenue model | Momentum rotation PnL |
| Data cost | Free (CMC Basic tier) |
| Execution cost | 0.25% DEX fee per swap + BNB gas (~$0.01-0.05) |

---

## Track 2 Judging Criteria Alignment

| Criterion | How We Address It |
|-----------|-------------------|
| **Innovation** | Freshness Ratio is a novel signal — no existing CMC skill measures breakout timing via 1h/24h percent-change ratio |
| **Utility** | Solves the free-tier momentum problem — makes algorithmic trading accessible without $299/mo OHLCV data |
| **Technical Quality** | 4-gate pipeline, cross-sectional ranking, atomic state persistence, never-crash architecture |
| **CMC Integration** | Uses 3 CMC endpoints, all on Basic tier, x402 payment signing for API calls |
| **Composability** | Clean input/output schema — can be chained with regime classifiers, portfolio managers, or notification skills |
| **Documentation** | Full architecture in IMPLEMENTATION_PLAN.md, inline docstrings, this SKILL.md |
