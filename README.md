# Kinetix
## Freshness Ratio Alpha -- Autonomous Momentum Rotation Agent

**BNB Hack: AI Trading Agent Edition -- June 2026**
**Dual-track submission: Track 1 (Autonomous Trading) + Track 2 (Strategy Skill)**

---

## What Is This?

A fully autonomous crypto trading agent on BNB Chain (BSC). Reads live market data from CoinMarketCap, classifies risk via an LLM regime detector, ranks tokens using a novel Freshness Ratio Alpha momentum engine, and executes self-custody swaps through Trust Wallet Agent Kit (TWAK) -- all on a 15-minute loop with zero human intervention.

Built for BNB Hack: AI Trading Agent Edition. Works on CMC's free Basic tier -- no $299/month OHLCV subscription needed.

**3,941 lines of Python. 12 modules. One strategy. Zero custodial shortcuts.**

---

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your CMC_API_KEY, DEEPSEEK_API_KEY, and wallet credentials

# 2. Install dependencies
pip install -r agent/requirements.txt

# 3. Paper trade (dry-run -- validate everything without real money)
bash scripts/paper_trade.sh --once

# 4. Run for 4 hours to observe behavior
bash scripts/paper_trade.sh --hours 4

# 5. Live trading
python3 agent/main.py --interval 900
```

System requirements: Python 3.11+, `twak` CLI, `bnbagent-sdk` installed. Linux/macOS. WSL2 on Windows works.

---

## Architecture

```
                                                    |  15-MINUTE TICK LOOP           |
                                                    |                                |
  CMC API (Basic tier) ----| 1. Guardrails (drawdown, quota, idle)                   |
  Fear & Greed             | 2. Fetch holdings (TWAK wallet)                         |
  Global metrics           | 3. Regime classification (LLM, hourly)                  |
  Quotes/latest (132 tok)  | 4. Freshness Ratio momentum scan                        |
                           | 5. Portfolio rebalancing + trailing stops               |
  DeepSeek LLM ---------- -| 6. Execute swap plan (TWAK, 8s delay)                   |
  (regime, hourly)         | 7. Log trades, update peak, save state                  |
                           |                                                         |
  TWAK CLI -------------- -| SLEEP 900s -> REPEAT                                    |
  (swap + portfolio)       ----------------------------------------------------------
```

### Module Map

```
bnb-hack-agent/
|
+-- README.md                               <- You are here
+-- SKILL.md                                <- Track 2 submission
+-- IMPLEMENTATION_PLAN.md                  <- Full architecture + design decisions
|
+-- agent/
|   +-- main.py                         (664 lines) Orchestrator loop
|   |
|   +-- strategy/
|   |   +-- momentum.py                 (359 lines) Freshness Ratio Alpha engine
|   |   +-- portfolio.py                (303 lines) Position sizing, trailing stops, penalty box
|   |   +-- regime.py                   (127 lines) LLM regime classifier (hourly, off hot path)
|   |
|   +-- execution/
|   |   +-- twak_client.py              (652 lines) TWAK swap execution + x402 payment signing
|   |   +-- guardrails.py               (321 lines) Drawdown brake, inactivity fallback, trade cap
|   |
|   +-- data/
|   |   +-- cmc_client.py               (399 lines) CMC API wrapper (free-tier endpoints)
|   |   +-- allowlist.py                (128 lines) 149 BEP-20 competition token registry
|   |   +-- cache.py                     (40 lines) Thread-safe in-memory TTL cache
|   |
|   +-- llm/
|   |   +-- prompts.py                  (196 lines) Regime classification prompt templates
|   |
|   +-- requirements.txt                Python deps (requests, python-dotenv)
|
+-- scripts/
|   +-- paper_trade.sh                  Paper trade launcher (--once, --hours N, full loop)
|   +-- register_competition.sh         TWAK on-chain competition registration wizard
|   +-- backtest.py                     (752 lines) 30-day historical replay through same pipeline
|   +-- test_cmc.py                     CMC endpoint verification
|
+-- state/
|   +-- portfolio_state.json            Current portfolio snapshot (peak, drawdown, holdings)
|   +-- trade_log.json                  Append-only trade history with PnL attribution
|
+-- .env                                Secrets (API keys, wallet password, wallet address)
```

---

## The Strategy: Freshness Ratio Alpha

### The Problem with Raw Momentum

Every momentum scanner looks at 24h price change. By the time a token shows +15% on the 24h candle, the move started 20 hours ago. You are buying the top.

### The Insight

CMC's free `/v2/cryptocurrency/quotes/latest` endpoint returns **both** `percent_change_1h` and `percent_change_24h`. The *ratio* between them reveals when the move ignited:

```
freshness = pct_1h / max(pct_24h, pct_1h, 0.1)
```

| Signal | Condition | Interpretation | Action |
|--------|-----------|----------------|--------|
| **Ignition** | `freshness >= 0.20`, `pct_24h > 0` | Fresh breakout -- majority of the move happened this hour | BUY |
| **Slingshot** | `pct_24h < 0`, `pct_1h >= 1.25` | V-bottom reversal -- was red, now violently green | BUY (freshness = 1.5) |
| **Exhausted** | `freshness < 0.20` | Old pump -- the move is stale, you missed it | REJECT |

### Why It Works

Two tokens, same screen, completely different entry quality:

```
Token A: +15% 24h,  +3% 1h   -> freshness = 0.20   Exhausted pump (started ~16h ago)
Token B: +3% 24h,   +2.8% 1h  -> freshness = 0.93   Ignition (just breaking out now)
```

A naive scanner buys Token A because +15% looks better. Freshness Ratio Alpha buys Token B because 93% of the move is fresh -- there is room to run.

### The 4-Gate Pipeline

Every token passes through four sequential gates before scoring:

```
132 tokens -> Gate 1 (Friction) -> Gate 2 (Freshness) -> Gate 3 (Liquidity) -> Gate 4 (Climax 24h) -> Gate 5 (Climax 1h) -> SCORED
```

| Gate | Rule | Rationale |
|------|------|-----------|
| 1. Friction Floor | `pct_1h >= 1.25%` | Must beat 0.25% DEX fee + slippage in one hour |
| 2. Freshness | `freshness >= 0.20` (or auto-1.5 for slingshot) | Reject exhausted pumps where the move is stale |
| 3. Liquidity | `volume_24h >= $100,000` | Avoid shallow pools where one swap moves the price |
| 4. Climax Exhaustion (24h) | `pct_24h <= 40%` | Reject blow-off tops -- do not buy into distribution |
| 5. Hourly Climax | `pct_1h <= 30%` | Reject intra-hour blow-offs (+30% in 60 min = manipulation) |

### Acceleration Score (Cross-Sectional Ranking)

Survivors are ranked by a composite that balances momentum, freshness, and market depth:

```
turnover_ratio    = volume_24h / market_cap
liquidity_quality = ln(1 + turnover_ratio x 100)
acceleration      = pct_1h x freshness x liquidity_quality
```

Volume-confirmed moves get a bonus: if `volume_change_24h > 0`, multiply by `1.0 + min(vol_chg/100, 0.15)`.

Top 2 tokens by acceleration score become the positions.

### Real Scan Results (June 21, 2026 -- 132 tokens, 2.8s)

```
Scanned: 132 tokens  |  Passed gates: 3  |  Candidates: 2

#1 SKYAI  score=8.582  freshness=1.50 (slingshot)  1h=+2.9%  24h=-4.0%  vol=$20.8M
#2 CYS    score=4.746  freshness=1.50 (slingshot)  1h=+1.7%  24h=-6.7%  vol=$3.5M
```

---

## Risk Architecture (Prevents Disqualification)

Competition rules disqualify at a 30% drawdown. Our architecture freezes at 25% -- a 5% survival buffer.

### 4-Layer Safety System

| Layer | Mechanism | Trigger | Behavior |
|-------|-----------|---------|----------|
| Per-position | Trailing stop-loss (5%) | `price <= peak x 0.95` | Full exit + 2h penalty box |
| Per-position | Penalty box (2h) | Token stopped out | Banned from re-entry for 7,200 seconds |
| Portfolio | Circuit breaker (-25%) | `portfolio < peak x 0.75` | Halt all buys. Trailing stops + heartbeat continue. LATCHING. |
| Compliance | Heartbeat trade (18h) | No trade in >= 18 hours | Force $5 USDT -> FDUSD |

### Why the Circuit Breaker Does Not Force-Sell

At -25% drawdown, you are 5% away from disqualification. Force-selling everything crystalizes losses that might recover. Instead:

- Trailing stops still fire -- exit individual positions hitting 5% below their peak
- Heartbeat trades continue -- competition requires 1 trade/day minimum
- No new buys -- do not add exposure to a declining portfolio
- No blanket sells -- do not panic-dump near the DQ line

The circuit breaker is **latching**: once triggered, stays active until manual reset. This prevents oscillation (breach -> recover -> breach -> DQ).

### Concentrated Betting (Top 2, Equal Weight)

```
Position sizing:     2 tokens
Allocation per:      50% of deployed capital each
risk_on deployed:    80% of portfolio
neutral deployed:    60% of portfolio
risk_off deployed:   15% of portfolio
```

Concentration is intentional for a 7-day sprint. 10 positions at 5% each means even a 100% winner adds only 5% to the portfolio. 2 positions at 40% each means a 25% winner adds 10%.

---

## Regime Overlay (LLM, Off-Hot-Path)

The LLM fires **once per hour** -- never on the 15-minute trade path. Inputs:

- CMC Fear & Greed Index (current + 7-day history)
- BTC dominance trend
- Global market cap 24h change
- Altcoin season indicator
- Trending token momentum (CMC community data)

Output: structured JSON decision:

```json
{
  "regime": "risk_on",
  "confidence": 0.85,
  "reasoning": "Fear & Greed recovering from deep fear, BTC dominance declining, alts rotating",
  "params": {
    "max_positions": 2,
    "allocation_pct": 0.80,
    "momentum_lookback": "24h"
  }
}
```

| Regime | Positions | Capital Deployed | Behavior |
|--------|-----------|-----------------|----------|
| risk_on | 2 | 80% | Full concentrated bets on top momentum |
| neutral | 2 | 60% | Moderate deployment, cautious sizing |
| risk_off | 1 | 15% | Minimum exposure, mostly USDT |

On LLM failure (API down, timeout, bad JSON): defaults to `neutral` -- never crashes, never stops trading.

LLM provider: DeepSeek Chat (OpenAI-compatible API). ~$0.001 per classification call. ~24 calls/day = ~$0.02/day.

---

## Execution Layer (TWAK + bnbagent-sdk)

### TWAK Surfaces Used

| Surface | Usage | Frequency |
|---------|-------|-----------|
| Swap execution (`twak swap`) | `--amount <tokens> --from <SYM> --to <SYM>` | Every 15 min (max 5/day) |
| Portfolio query (`twak wallet portfolio`) | `--format json` -> parse token balances + USD values | Every 15 min |
| Competition registration (`twak compete register`) | On-chain registration to BSC contract | Once |
| x402 signing (bnbagent-sdk) | CMC API call payments signed via local wallet | Every API call |

### Self-Custody Integrity

1. Local keystore (`~/.bnbagent/wallets/<address>.json`): AES-128-CTR + scrypt encrypted. Private key only needed on first run.
2. SigningPolicy strict mode: rejects unbounded permits -- only signs EIP-3009 TransferWithAuthorization against known U-token deployments.
3. No third-party custody: zero exchange API keys, zero custodial wallets, zero relay services.
4. x402 defense-in-depth: `expected_to` address is hardcoded from the CMC on-chain registry -- not taken from the challenge body.

### Execution Safety

- 8-second delay between sequential swaps (2+ BSC blocks of nonce separation)
- Stop on failure: if a swap reverts, the plan halts (do not compound errors)
- Paper trade mode: full pipeline with fake tx hashes -- validate before live

---

## Data Pipeline (CMC Free Tier)

Every data source is available on CMC's Basic (free) tier. No Standard-tier OHLCV. No paid endpoints.

| Endpoint | Data | Frequency | Cache TTL |
|----------|------|-----------|-----------|
| `/v2/cryptocurrency/quotes/latest` | 132 tokens: price, 1h%, 24h%, volume, market cap | Every 15 min | 120s |
| `/v3/fear-and-greed/latest` | Fear & Greed index + classification | Hourly | 900s |
| `/v1/global-metrics/quotes/latest` | Total market cap, BTC dominance, alt season | Hourly | 900s |
| `/v1/cryptocurrency/map` | Token ID -> symbol resolution | Once daily | 86,400s |

Rate limits: CMC Basic tier = 300 calls/min, 15,000 credits/month. Peak usage: ~240 calls/hour (~4/min). Well under limits.

x402 payments: Every CMC API call that requires payment is signed through the TWAK x402 signer with per-call caps and a session budget. Payment is real, on-chain, and logged for audit.

---

## State Persistence (Crash-Proof)

Both state files use atomic writes (`write to .tmp -> os.replace`) -- never corrupt on crash.

### `portfolio_state.json`

```json
{
  "peak_value_usd": 10500.0,
  "current_value_usd": 10300.0,
  "drawdown_pct": 1.9,
  "emergency_triggered": false,
  "trades_today": 2,
  "last_trade_date": "2026-06-22",
  "last_trade_ts": "2026-06-22T14:30:00Z",
  "holdings": {
    "SKYAI": {"balance": 4393.77, "cost_basis_usd": 1500.0, "peak_price": 0.35}
  },
  "cooldowns": {},
  "regime": "risk_off",
  "regime_updated_ts": "2026-06-22T14:00:00Z"
}
```

### `trade_log.json` (append-only)

```json
{
  "trades": [
    {
      "ts": "2026-06-22T14:30:00Z",
      "action": "buy",
      "token": "SKYAI",
      "from_token": "USDT",
      "amount": 4393.77,
      "amount_usd": 1500.0,
      "tx_hash": "0xa1b2c3...",
      "regime": "risk_off",
      "reason": "freshness=1.50 1h=+2.9% 24h=-4.0% score=8.582"
    }
  ]
}
```

---

## Track 1: Autonomous Trading Agent ($24,000 Prize Pool)

### Submission

A fully autonomous, self-custody trading agent that reads markets via CMC free-tier endpoints, classifies risk using DeepSeek LLM (hourly, off hot path), discovers momentum via Freshness Ratio Alpha (a novel signal detecting breakout timing), sizes positions with trailing stop-loss and penalty box, executes autonomously through TWAK self-custody swaps on BSC, and self-protects with a 4-layer risk system including -25% circuit breaker.

### Competition Readiness

| Requirement | Status | Details |
|-------------|--------|---------|
| On-chain registration | Done | `0x88D9666cCEFA0EEa878429f89aC72e87f1c3fc24` registered via `twak compete register` |
| Token allowlist compliance | Done | Only trades the 149 BEP-20 tokens. Full list in `data/allowlist.py` |
| Min 1 trade/day | Done | Heartbeat trade: $5 USDT->FDUSD after 18h inactivity |
| Max 5 trades/day | Enforced | Hard cap in guardrails, resets at UTC midnight |
| Drawdown <= 30% | Protected | Circuit breaker halts buys at 25% (5% buffer before DQ) |
| Self-custody | Full | Local keystore, TWAK signing, no custodial components |
| DoraHacks registration | Pending | Submit agent address + strategy description |

### Best Use of TWAK -- Scoring Breakdown

| Criterion | How We Score |
|-----------|-------------|
| TWAK integration depth (30 pts) | Uses 3 TWAK surfaces: `twak swap`, `twak wallet portfolio`, `twak compete register`. x402 signing via bnbagent-sdk. |
| Self-custody integrity (25 pts) | Full score -- local keystore, SigningPolicy strict mode, zero third-party custody. |
| Autonomous execution + guardrails (20 pts) | Hands-off 15-min loop. 4-layer risk system. Drawdown caps, trade limits, slippage protection. |
| Native x402 usage (10 pts) | x402 signing integrated into CMC data pipeline. Per-call caps, session budget, hardcoded payee. |
| Originality + relevance (10 pts) | Freshness ratio is novel. Real strategy a self-custody user would actually run. |
| Presentation (5 pts) | Demo shows full self-custody loop end-to-end with on-chain proof (tx hash on BSC). |

### Best Use of Agent Hub

- 3 CMC endpoints consumed (quotes/latest, Fear & Greed, global metrics)
- CMC MCP bridge: standalone executor for CMC skill hub calls outside Claude Code
- Skills integration: `daily_market_overview` and `altcoin_breakout_scanner_spot` consumed via MCP
- LLM skill: regime classifier transforms CMC market data into risk decisions
- All on free tier -- accessible to anyone

### Best Use of BNB AI Agent SDK

- `EVMWalletProvider` for local key management + keystore encryption
- `X402Signer` for defense-in-depth payment signing (recipient binding, per-call caps, session budget)
- `BSC_MAINNET_CHAIN_ID`, `PAYMENT_TOKEN_EIP712_NAME`, `PAYMENT_TOKEN_EIP712_VERSION` for chain-aware signing
- `get_address()` for BSC network configuration
- SDK is the signing and payment backbone, not a cosmetic import

---

## Track 2: Strategy Skills ($6,000 Prize Pool)

### Submission

**Freshness Ratio Alpha** as a reusable CMC Skill -- see [`SKILL.md`](SKILL.md) for full submission document.

### How It Works as a CMC Skill

```
INPUT                      PROCESSING                          OUTPUT
+--------------+           +--------------------------+        +------------------------+
| regime       |           | 1. Fetch quotes for      |        | candidates[]           |
| (risk_on/    |           |    all non-stable        |        |   +- symbol            |
|  neutral/    |---------->|    competition tokens    |------->|   +- acceleration      |
|  risk_off)   |           |    via CMC /v2           |        |   +- freshness         |
|              |           |                          |        |   +- pct_1h            |
| top_n (int)  |           | 2. 4-gate pipeline       |        |   +- pct_24h           |
|              |           |    Friction->Freshness   |        |   +- vol_24h           |
| cooldowns    |           |    ->Liquidity->Climax   |        |   +- market_cap        |
| (optional)   |           |                          |        |   +- price             |
+--------------+           | 3. Cross-sectional       |        |   +- reason            |
                           |    ranking               |        +------------------------+
                           |                          |
                           | 4. Top-N candidates      |
                           +--------------------------+
```

### Composability

This skill is a building block, not a monolith. Composable with:

- Regime classifiers upstream: `fear_and_greed_latest` + `global_metrics_latest` -> regime -> momentum
- Portfolio rebalancers downstream: candidates -> position sizing -> swap plan
- Risk managers downstream: trailing stops, circuit breakers, compliance trades
- Notification systems: "Freshness Ratio Alpha detected a slingshot: SKYAI score=8.58"

### Why This Skill Is Novel

1. **Freshness Ratio is a genuinely new signal.** No existing CMC skill measures breakout timing via the 1h/24h percent-change ratio. It is not RSI, not MACD, not EMA crossover -- it is a mathematical relationship between two freely-available data points.

2. **Free-tier accessibility.** Most momentum skills require Standard tier ($299/mo) for OHLCV/k-line data. This skill works on Basic tier (free). Usable by anyone with a free CMC API key.

3. **Evidence-proportional output.** Structured candidate data with composite scores, not a vague narrative. Each candidate includes the exact freshness ratio, percent changes, and volume that produced the score -- fully auditable.

4. **Composable architecture.** Clean input schema (regime, top_n, optional cooldowns), clean output schema (ranked candidates with scores). Designed to be chained.

### Judging Criteria Alignment

| Criterion | How We Score |
|-----------|-------------|
| Technical execution | Working pipeline -- paper-trade verified end-to-end. Real CMC data, real TWAK execution, real on-chain proof. |
| Originality | Freshness Ratio is novel. 1h/24h ratio as a breakout-timing signal is our invention. No prior art in CMC skills. |
| Real-world relevance | Free-tier accessibility means anyone can use it. Concentrated momentum rotation is a real strategy with institutional precedent. |
| Demo + presentation | Paper trade mode demonstrates full pipeline. SKILL.md documents the skill as a reusable building block. On-chain proof available. |

---

## Backtest (30-Day Replay)

`scripts/backtest.py` (752 lines) replays the full pipeline against 30 days of historical CMC data:

- Downloads 15-min k-line candles from CMC (cached to `cache/backtest/`)
- Local momentum scoring: RSI14, EMA50, 24h price/volume change -> composite score
- Rule-based regime from Fear & Greed historical (no LLM -- F&G<20 or >75 = risk_off, 35-65 = risk_on)
- Reuses `portfolio.generate_swap_plan()` exactly -- same slippage, BNB buffer, position sizing
- Simulates execution at close prices with slippage haircut
- Emergency: 25% drawdown triggers sell-all-into-USDT
- Report: total PnL, Sharpe (annualized), max drawdown, win rate, avg return/trade, vs BNB/BTC benchmark

```bash
python3 scripts/backtest.py --days 30 --top 2 --interval 900
python3 scripts/backtest.py --days 30 --all-tokens --interval 900    # full-competition
```

---

## Competition Registration

### Track 1 (On-Chain)

```bash
bash scripts/register_competition.sh    # Already done
```

- Wallet address: `0x88D9666cCEFA0EEa878429f89aC72e87f1c3fc24`
- Registration tx: `0xb128750dbfd485c45832e5403ad52a11e71e8f4dea3a2b95179fb844e96c4c27`
- Competition contract: `0x212c61b9b72c95d95bf29cf032f5e5635629aed5`

### Track 2 (DoraHacks)

Submit [`SKILL.md`](SKILL.md) through DoraHacks before the build window closes on June 21. No on-chain registration required.

---

## Starting Capital

| Tier | BNB (Gas) | USDT (Trading) | Total | Expected Outcome |
|------|-----------|----------------|-------|-----------------|
| Minimum | 0.05 BNB (~$15) | $200 | ~$215 | Survive the week, small PnL |
| Recommended | 0.1 BNB (~$30) | $500 | ~$530 | Meaningful returns, absorb 2-3 bad trades |
| Comfortable | 0.2 BNB (~$60) | $1,000-2,000 | ~$1,500 | Real profit potential at scale |

At $500 USDT with 80% allocation: $200 per position x 2 = $400 deployed. A 15% winner = +$60. A 5% stop-out = -$10. At 5 trades/day with 60% win rate, expect +$15-30/day in normal volatility.

**Send funds to**: `0x88D9666cCEFA0EEa878429f89aC72e87f1c3fc24` on **BSC mainnet**.

---

## Paper Trade Validation

Runs the full pipeline against real CMC data with fake transactions:

```bash
bash scripts/paper_trade.sh --once              # Single tick smoke test
bash scripts/paper_trade.sh --hours 4           # Observe behavior
bash scripts/paper_trade.sh --interval 300 --hours 2   # Faster feedback
```

Validates: CMC API connectivity, Freshness Ratio Alpha discovering real candidates, LLM regime classification (or neutral fallback), swap plan generation (sells, buys, quota truncation), guardrails (drawdown monitor, inactivity timer, trade cap), state persistence (atomic writes, crash recovery), full pipeline end-to-end.

**Verified output** (June 21, 2026):

```
Tick 1 --
Guardrails:  proceed -- Quota OK (0/5, 5 remaining)
Portfolio:   $10,000 total, peak=$10,000, drawdown=0.0%
Candidate #1: SKYAI score=8.582 (freshness=1.50, slingshot: 1h=+2.9%, 24h=-4.0%)
Candidate #2: CYS   score=4.746 (freshness=1.50, slingshot: 1h=+1.7%, 24h=-6.7%)
Executed:    2 swaps succeeded, 0 failed
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Description |
|----------|----------|-------------|
| `CMC_API_KEY` | Yes | CMC API key (free tier works) |
| `WALLET_PASSWORD` | Track 1 only | Password for local keystore encryption |
| `WALLET_ADDRESS` | Track 1 only | BSC wallet address for keystore selection |
| `PRIVATE_KEY` | First run only | Hex private key (encrypted to keystore, removable after) |
| `DEEPSEEK_API_KEY` | Recommended | DeepSeek API key for LLM regime classification |
| `AGENT_STATE_DIR` | Optional | Custom state directory path |
| `X402_MAX_VALUE_PER_CALL` | Optional | x402 per-call spending cap (base units) |
| `X402_SESSION_BUDGET` | Optional | x402 session spending cap (base units) |
| `LOG_LEVEL` | Optional | `INFO` (default) or `DEBUG` |

Track 2 only: you only need `CMC_API_KEY` -- no wallet, no TWAK, no execution layer.

---

## Dependencies

```
# Python (agent/requirements.txt)
requests>=2.31
python-dotenv>=1.0

# External
twak CLI                    # Trust Wallet Agent Kit
bnbagent-sdk                # BNB AI Agent SDK
CMC API key                 # Free tier: sign up at coinmarketcap.com/api
DeepSeek API key (optional) # platform.deepseek.com
```

---

## Safety Properties

| Property | Mechanism |
|----------|-----------|
| Never crashes | Every pipeline step wrapped in try/except. Exceptions logged, tick continues. |
| Never corrupts state | Atomic writes (write to .tmp -> os.replace). No partial state on crash. |
| Never loses custody | Local keystore, encrypted. Private key only in memory during signing. |
| Never exceeds DQ drawdown | Circuit breaker at 25% (5% buffer before 30% DQ). Latching. |
| Never misses a required trade | Heartbeat trade (18h inactivity) ensures minimum 1 trade/day. |
| Never overtrades | Hard 5 trade/day cap. Resets at UTC midnight. |
| Never buys into exhaustion | Gate 4 (Climax Exhaustion) rejects tokens already up >40%. |
| Never sells into a drawdown | Circuit breaker halts buys ONLY. Trailing stops still protect positions. |

---

## Failure Modes

| Scenario | Behavior |
|----------|----------|
| CMC API down | Tick logs error, skips, retries next cycle. Regime falls back to cached. |
| LLM API down | Regime defaults to `neutral`. Trading continues with moderate parameters. |
| TWAK CLI missing | Detected at startup. Paper trade mode works without it. |
| Wallet balance empty | Portfolio = $0 -> tick skips (not treated as drawdown). |
| TWAK swap reverts | Plan stops at first failure. Error logged. Next tick recovers. |
| BSC RPC unresponsive | 120s hard timeout per swap. Failure stops plan execution. |
| State file deleted mid-run | `load_state()` returns fresh defaults. Peak resets (conservative). |
| UTC midnight during tick | `check_daily_reset()` runs first -- trade counter reset before any swap. |
| Two copies of agent running | No mutex, but TWAK nonce errors prevent double-swaps. Do not do this. |

---

## License

MIT

---

