#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# BNB Hack Competition Registration
# ═══════════════════════════════════════════════════════════════════════════
# Registers your agent wallet with the BNB Hack competition contract and
# does a pre-flight check on everything needed for live trading.
#
# Usage:
#   bash scripts/register_competition.sh           # guided registration
#   bash scripts/register_competition.sh --check   # pre-flight only
#
# Registration contract (Track 1 — Autonomous Trading Agents):
#   0x212c61b9b72c95d95bf29cf032f5e5635629aed5
#
# What this script does:
#   1. Check TWAK CLI is installed
#   2. Check environment: WALLET_PASSWORD, CMC_API_KEY
#   3. Verify wallet is accessible and has BNB for gas
#   4. Register via: twak compete register
#   5. Print DoraHacks registration reminder
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPETITION_CONTRACT="0x212c61b9b72c95d95bf29cf032f5e5635629aed5"

# ── Colors ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✅ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
err()  { echo -e "${RED}❌ $1${NC}"; }
info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# ── Check mode only ───────────────────────────────────────────────────────
CHECK_ONLY=false
if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=true
fi

echo "═══════════════════════════════════════════════════════════"
echo "  BNB Hack Track 1 — Competition Registration"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: TWAK CLI
# ═══════════════════════════════════════════════════════════════════════════
echo "─── Step 1: TWAK CLI ───"

if command -v twak &>/dev/null; then
    TWAK_VER=$(twak --version 2>&1 | head -1 || echo "unknown")
    ok "TWAK CLI found: $TWAK_VER"
    TWA_BIN="twak"
elif command -v npx &>/dev/null && npx twak --version &>/dev/null 2>&1; then
    ok "TWAK CLI via npx"
    TWA_BIN="npx twak"
else
    err "TWAK CLI not found."
    info "Install with: npm install -g @trustwallet/cli"
    info "See: https://docs.twak.io/cli/install"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Environment
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "─── Step 2: Environment ───"

ENV_FILE="$PROJECT_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    err ".env file not found at $ENV_FILE"
    info "Copy the template: cp .env.example .env"
    info "Then fill in your values: WALLET_PASSWORD, CMC_API_KEY, etc."
    exit 1
fi

# Source .env (export variables)
set -a
source "$ENV_FILE"
set +a

# Validate required vars
MISSING=()
for var in WALLET_PASSWORD CMC_API_KEY; do
    if [[ -z "${!var:-}" ]]; then
        MISSING+=("$var")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    err "Missing required env vars in .env: ${MISSING[*]}"
    exit 1
fi
ok ".env file OK"

if [[ -n "${PRIVATE_KEY:-}" ]]; then
    info "PRIVATE_KEY set — first run will import and encrypt wallet"
elif [[ -n "${WALLET_ADDRESS:-}" ]]; then
    info "WALLET_ADDRESS=$WALLET_ADDRESS — loading existing keystore"
else
    info "No PRIVATE_KEY or WALLET_ADDRESS — will try auto-select if keystore exists"
fi

if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
    ok "DEEPSEEK_API_KEY set — regime classification active"
else
    warn "DEEPSEEK_API_KEY not set — agent runs in neutral-only mode"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Wallet verification
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "─── Step 3: Wallet ───"

# Try to get wallet address via TWAK
WALLET_ADDR=$($TWA_BIN wallet address 2>/dev/null || echo "")
if [[ -z "$WALLET_ADDR" ]]; then
    warn "Could not resolve wallet address via TWAK."
    info "Make sure wallet is initialized: twak wallet init"
else
    ok "Wallet address: $WALLET_ADDR"

    # Check BNB balance (best-effort — TWAK may not support this directly)
    if command -v curl &>/dev/null; then
        BNB_BALANCE_WEI=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -d "{\"jsonrpc\":\"2.0\",\"method\":\"eth_getBalance\",\"params\":[\"$WALLET_ADDR\",\"latest\"],\"id\":1}" \
            "https://bsc-dataseed1.binance.org" 2>/dev/null | \
            python3 -c "import sys,json; print(int(json.load(sys.stdin)['result'],16))" 2>/dev/null || echo "0")

        BNB_BALANCE=$(python3 -c "print(${BNB_BALANCE_WEI:-0} / 1e18)" 2>/dev/null || echo "0")
        if python3 -c "exit(0 if ${BNB_BALANCE} >= 0.003 else 1)" 2>/dev/null; then
            ok "BNB balance: ${BNB_BALANCE} BNB (≥0.003, gas OK)"
        else
            warn "BNB balance: ${BNB_BALANCE} BNB — may need more for gas"
            info "Minimum ~0.003 BNB for gas during competition week"
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Register
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "─── Step 4: Registration ───"

if $CHECK_ONLY; then
    info "--check mode — skipping registration tx"
else
    info "Registering with competition contract: $COMPETITION_CONTRACT"
    echo ""
    echo "Running: twak compete register"
    echo "Contract: $COMPETITION_CONTRACT"
    echo ""

    read -p "Proceed? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "Skipped registration."
    else
        $TWA_BIN compete register --contract "$COMPETITION_CONTRACT" 2>&1 || {
            err "Registration failed. Check TWAK docs or run with --check first."
            info "Manual registration: $TWA_BIN compete register"
            exit 1
        }
        ok "Registration submitted!"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 5: DoraHacks reminder
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "─── Step 5: DoraHacks ───"
echo ""
echo "⚠️  ALSO register on DoraHacks:"
echo "   https://dorahacks.io/hackathon/bnb-hack-2026"
echo ""
echo "   You'll need:"
echo "   - Your agent wallet address: ${WALLET_ADDR:-<unknown>}"
echo "   - Strategy description (see BNB_HACK_SUBMISSION.md)"
echo "   - GitHub repo link"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "  Registration Checklist"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  [ ] TWAK CLI installed"
echo "  [ ] .env configured (WALLET_PASSWORD, CMC_API_KEY)"
echo "  [ ] Wallet funded with BNB for gas"
if $CHECK_ONLY; then
    echo "  [ ] Registration tx (run without --check)"
else
    echo "  [ ] Registration tx submitted or skipped"
fi
echo "  [ ] DoraHacks registration"
echo "  [ ] Paper trade dry-run (run: bash scripts/paper_trade.sh)"
echo ""
echo "═══════════════════════════════════════════════════════════"

if $CHECK_ONLY; then
    ok "Pre-flight check complete."
else
    ok "Registration script complete."
fi
