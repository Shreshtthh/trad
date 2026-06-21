#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Paper Trade Dry-Run
# ═══════════════════════════════════════════════════════════════════════════
# Runs the full orchestrator in paper-trade mode — fake transactions,
# real market data. Validates the entire pipeline before live trading.
#
# Usage:
#   bash scripts/paper_trade.sh              # one tick, 15-min loop
#   bash scripts/paper_trade.sh --once       # single tick then exit
#   bash scripts/paper_trade.sh --hours 4    # run for 4 hours
#
# What this validates:
#   - CMC API connectivity (regime, momentum, prices)
#   - LLM regime classification (or neutral fallback)
#   - Portfolio swap plan generation
#   - Guardrails (drawdown, inactivity, quota)
#   - State persistence (portfolio_state.json, trade_log.json)
#   - Full pipeline end-to-end
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Colors ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✅ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
err()  { echo -e "${RED}❌ $1${NC}"; }
info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# ── Args ──
ONCE=false
RUNTIME_HOURS=""
INTERVAL=900

while [[ $# -gt 0 ]]; do
    case "$1" in
        --once) ONCE=true; shift ;;
        --hours) RUNTIME_HOURS="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

echo "═══════════════════════════════════════════════════════════"
echo "  Paper Trade Dry-Run"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Pre-flight ────────────────────────────────────────────────────────────

# 1. .env
ENV_FILE="$PROJECT_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    err ".env not found at $ENV_FILE"
    info "Run: cp .env.example .env  and fill in values"
    exit 1
fi

# Export all vars from .env so Python subprocess can see them
set -a
source "$ENV_FILE"
set +a

# 2. Python venv
PYTHON=""
for candidate in \
    "$PROJECT_DIR/venv/bin/python3" \
    "$PROJECT_DIR/.venv/bin/python3" \
    "/root/trad/bnbagent-sdk/.venv/bin/python3" \
    "python3"; do
    if $candidate -c "import sys; sys.path.insert(0,'$PROJECT_DIR/agent'); from strategy.portfolio import generate_swap_plan" 2>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    err "No working Python with agent dependencies found."
    info "Create a venv:   python3 -m venv venv && source venv/bin/activate"
    info "Then:            pip install -r agent/requirements.txt"
    exit 1
fi
ok "Python: $PYTHON"

# 3. CMC API key (already sourced with set -a above)
if [[ -z "${CMC_API_KEY:-}" ]]; then
    err "CMC_API_KEY not set in .env"
    exit 1
fi
ok "CMC_API_KEY: set"

# 4. Optional: DeepSeek
if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
    ok "DEEPSEEK_API_KEY: set (regime classification active)"
else
    warn "DEEPSEEK_API_KEY not set (neutral-only regime)"
fi

echo ""

# ── Run ───────────────────────────────────────────────────────────────────

cd "$PROJECT_DIR"

if $ONCE; then
    info "Running single tick (paper trade)..."
    echo ""

    $PYTHON -u agent/main.py \
        --paper-trade \
        --once \
        --twak-bin "echo" \
        2>&1 | grep -v "^$\|^\[DEPRECATED\]" || true

    echo ""
    ok "Single tick complete."

    # Show state
    STATE_FILE="$PROJECT_DIR/state/portfolio_state.json"
    if [[ -f "$STATE_FILE" ]]; then
        echo ""
        info "State after tick:"
        $PYTHON -c "
import json
with open('$STATE_FILE') as f:
    s = json.load(f)
print(f'  peak_value_usd:  \${s[\"peak_value_usd\"]:,.0f}')
print(f'  drawdown_pct:    {s[\"drawdown_pct\"]}%')
print(f'  trades_today:    {s[\"trades_today\"]}/5')
print(f'  regime:          {s[\"regime\"]}')
print(f'  emergency:       {s[\"emergency_triggered\"]}')
"
    fi

    TRADE_LOG="$PROJECT_DIR/state/trade_log.json"
    if [[ -f "$TRADE_LOG" ]]; then
        TRADE_COUNT=$($PYTHON -c "
import json
with open('$TRADE_LOG') as f:
    log = json.load(f)
print(len(log.get('trades', [])))
" 2>/dev/null || echo "0")
        echo "  trades logged:   $TRADE_COUNT"
    fi

else
    if [[ -n "$RUNTIME_HOURS" ]]; then
        SECONDS=$((RUNTIME_HOURS * 3600))
        info "Running paper trade for ${RUNTIME_HOURS}h (interval=${INTERVAL}s)..."
        info "Press Ctrl+C to stop early."
        echo ""

        timeout "$SECONDS" $PYTHON -u agent/main.py \
            --paper-trade \
            --interval "$INTERVAL" \
            --twak-bin "echo" \
            2>&1 || true
    else
        info "Running paper trade loop (interval=${INTERVAL}s)..."
        info "Press Ctrl+C to stop."
        echo ""

        $PYTHON -u agent/main.py \
            --paper-trade \
            --interval "$INTERVAL" \
            --twak-bin "echo" \
            2>&1
    fi
fi

echo ""
ok "Paper trade run complete."
echo ""
echo "Check results:"
echo "  State:  cat state/portfolio_state.json | python3 -m json.tool"
echo "  Trades: cat state/trade_log.json | python3 -m json.tool"
