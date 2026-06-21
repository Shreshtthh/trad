"""
TWAK execution wrapper — bridges the swap plan to on-chain execution.

Uses bnbagent-sdk for wallet management and x402 payment signing;
subprocess calls to the TWAK CLI for swap execution and portfolio queries.

Architecture:
  ┌─────────────┐     ┌──────────────┐     ┌───────────┐
  │ portfolio.py │ ──▶ │ twak_client  │ ──▶ │ TWAK CLI  │ ──▶ BSC chain
  │ SwapPlan     │     │ execute_swap │     │ twak swap │
  └─────────────┘     │ fetch_hold.. │     │ twak...   │
                      │ sign_x402    │     └───────────┘
                      └──────┬───────┘
                             │
                      ┌──────▼───────┐
                      │ bnbagent-sdk │
                      │ EVMWallet    │
                      │ X402Signer   │
                      └──────────────┘
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from bnbagent import EVMWalletProvider, X402Signer
from bnbagent.networks import (
    BSC_MAINNET_CHAIN_ID,
    PAYMENT_TOKEN_EIP712_NAME,
    PAYMENT_TOKEN_EIP712_VERSION,
    get_address,
)
from bnbagent.x402 import (
    X402RecipientMismatchError,
    X402AmountExceededError,
    X402BudgetExhaustedError,
    X402PolicyError,
)

log = logging.getLogger(__name__)

# ── Competition constants ──
MAX_TRADES_PER_DAY = 5
MIN_TRADE_USD = 5.0
BNB_GAS_BUFFER_USD = 20.0

# ── Execution safety ──
SWAP_DELAY_SECONDS = 8       # delay between sequential swaps (nonce safety)
SWAP_TIMEOUT_SECONDS = 120   # hard timeout per swap (RPC stall protection)
FETCH_TIMEOUT_SECONDS = 30   # hard timeout for portfolio fetch

# ── x402 payment limits (CMC API calls) ──
# Values in raw base units of the BSC mainnet U-token (6 decimals).
# 1 USDC-equivalent = 1_000_000 base units.
CMC_MAX_VALUE_PER_CALL = 1_000_000    # 1 U per API call max
CMC_SESSION_BUDGET = 50_000_000       # 50 U total session budget

# ── EIP-712 schema for TransferWithAuthorization (x402) ──
EIP712_DOMAIN_FIELDS = [
    {"name": "name", "type": "string"},
    {"name": "version", "type": "string"},
    {"name": "chainId", "type": "uint256"},
    {"name": "verifyingContract", "type": "address"},
]

TWA_FIELDS = [
    {"name": "from", "type": "address"},
    {"name": "to", "type": "address"},
    {"name": "value", "type": "uint256"},
    {"name": "validAfter", "type": "uint256"},
    {"name": "validBefore", "type": "uint256"},
    {"name": "nonce", "type": "bytes32"},
]


@dataclass
class TradeResult:
    """Result of a single TWAK swap execution."""
    success: bool
    tx_hash: Optional[str] = None
    from_token: str = ""
    to_token: str = ""
    amount_token: float = 0.0
    error: Optional[str] = None


@dataclass
class Holdings:
    """Parsed wallet holdings from TWAK."""
    tokens: dict[str, dict] = field(default_factory=dict)
    total_value_usd: float = 0.0
    raw_output: str = ""


class TwakClient:
    """
    Bridge between the agent's swap plan and on-chain execution via TWAK.

    Combines:
      - bnbagent-sdk EVMWalletProvider for key management + x402 signing
      - TWAK CLI (subprocess) for swap execution + portfolio queries

    Usage::

        client = TwakClient.from_env()
        holdings = client.fetch_holdings()
        result = client.execute_swap(swap_instruction)
        payment = client.sign_x402_payment(challenge_dict, expected_payee)
    """

    def __init__(
        self,
        wallet: EVMWalletProvider,
        x402_signer: X402Signer,
        *,
        twak_bin: str = "twak",
        paper_trade: bool = False,
    ) -> None:
        self._wallet = wallet
        self._x402 = x402_signer
        self._twak_bin = twak_bin
        self._paper_trade = paper_trade

        # Verify twak CLI is reachable
        if not paper_trade and not self._twak_found():
            log.warning("twak CLI not found at %r — swap execution will fail", twak_bin)

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_env(
        cls,
        *,
        twak_bin: str = "twak",
        paper_trade: bool = False,
    ) -> "TwakClient":
        """
        Create a TwakClient from environment variables.

        Required env vars:
          - WALLET_PASSWORD: password for keystore encryption
          - PRIVATE_KEY: hex private key (only needed on first run;
            after that the keystore is loaded from ~/.bnbagent/wallets/)

        Optional:
          - WALLET_ADDRESS: specify which keystore to load (when multiple exist)
          - X402_MAX_VALUE_PER_CALL: override per-call cap (base units)
          - X402_SESSION_BUDGET: override session budget (base units)

        Paper-trade shortcut: when paper_trade=True and no PRIVATE_KEY /
        WALLET_ADDRESS is set, wallet + x402 init is skipped entirely.
        The paper wallet address defaults to WALLET_ADDRESS env var, or
        a placeholder for display purposes.
        """
        if paper_trade:
            return cls._paper_from_env(twak_bin=twak_bin)

        password = os.getenv("WALLET_PASSWORD")
        if not password:
            raise ValueError("WALLET_PASSWORD env var is required")

        private_key = os.getenv("PRIVATE_KEY") or None
        address = os.getenv("WALLET_ADDRESS") or None

        # If no private key and no address, try auto-select (works if exactly one keystore exists)
        if not private_key and not address:
            if EVMWalletProvider.keystore_exists():
                log.info("No PRIVATE_KEY or WALLET_ADDRESS set — auto-selecting sole keystore")
            else:
                raise ValueError(
                    "No PRIVATE_KEY set and no keystore found in ~/.bnbagent/wallets/. "
                    "Set PRIVATE_KEY on first run to import and encrypt the wallet."
                )

        wallet = EVMWalletProvider(
            password=password,
            private_key=private_key,
            address=address,
        )
        log.info("Wallet loaded: %s (source=%s)", wallet.address, wallet.source)

        # Payment token for BSC mainnet
        bsc_mainnet = get_address(BSC_MAINNET_CHAIN_ID)
        payment_token = bsc_mainnet.payment_token

        max_per_call = int(os.getenv("X402_MAX_VALUE_PER_CALL", str(CMC_MAX_VALUE_PER_CALL)))
        session_budget = int(os.getenv("X402_SESSION_BUDGET", str(CMC_SESSION_BUDGET)))

        x402_signer = X402Signer(
            wallet,
            max_value_per_call={payment_token: max_per_call},
            session_budget={payment_token: session_budget},
        )
        log.info(
            "X402Signer ready: max_per_call=%d, session_budget=%d, token=%s",
            max_per_call, session_budget, payment_token,
        )

        return cls(wallet, x402_signer, twak_bin=twak_bin, paper_trade=paper_trade)

    @classmethod
    def _paper_from_env(cls, *, twak_bin: str = "twak") -> "TwakClient":
        """Create a paper-trade client with no real wallet."""
        address = os.getenv("WALLET_ADDRESS") or "0x0000000000000000000000000000000000000000"
        log.info("Paper trade: no real wallet — display address=%s", address[:10] + "...")

        # Paper wallet stub with just an address attribute
        paper_wallet = type("_PaperWallet", (), {"address": address})()
        paper_x402 = type("_PaperX402", (), {
            "budget": type("_Budget", (), {"spent": lambda self, token: 0})(),
        })()
        return cls(paper_wallet, paper_x402, twak_bin=twak_bin, paper_trade=True)

    # ── Public properties ────────────────────────────────────────────────

    @property
    def wallet_address(self) -> str:
        return self._wallet.address

    @property
    def x402_budget_spent(self, token: str | None = None) -> int:
        """Total base units spent through x402 this session."""
        if token is None:
            token = get_address(BSC_MAINNET_CHAIN_ID).payment_token
        return self._x402.budget.spent(token)

    # ── TWAK CLI helpers ─────────────────────────────────────────────────

    def _twak_found(self) -> bool:
        """Check if the TWAK CLI binary is reachable."""
        try:
            result = subprocess.run(
                [self._twak_bin, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _run_twak(self, args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
        """Run a TWAK CLI command. Raises RuntimeError on failure."""
        cmd = [self._twak_bin] + args
        log.info("TWAK: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"TWAK CLI binary {self._twak_bin!r} not found. "
                f"Install TWAK or set paper_trade=True for offline mode."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"TWAK command timed out after {timeout}s: {' '.join(cmd)}")

        if result.returncode != 0:
            raise RuntimeError(
                f"TWAK command failed (exit={result.returncode}): "
                f"{' '.join(cmd)}\nstderr: {result.stderr.strip()}"
            )
        return result

    # ── Swap execution ───────────────────────────────────────────────────

    @staticmethod
    def _get_field(obj, field: str, default=None):
        """Read field from a dict or attribute-carrying object."""
        if isinstance(obj, dict):
            return obj.get(field, default)
        return getattr(obj, field, default)

    def execute_swap(self, instruction) -> TradeResult:
        """
        Execute a single swap via TWAK CLI.

        Args:
            instruction: SwapInstruction (object) OR plain dict with keys
                action, from_token, to_token, amount_token, amount_usd.
                Dict support allows emergency/compliance swaps to bypass
                the portfolio SwapInstruction code path.

        Returns:
            TradeResult with tx_hash on success.

        The TWAK CLI command is:
            twak swap --amount <amount_token> --from <from_token> --to <to_token>
        """
        _ = self._get_field  # short alias
        action = _(instruction, "action")
        from_tok = _(instruction, "from_token")
        to_tok = _(instruction, "to_token")
        amount = _(instruction, "amount_token", 0)
        amount_usd = _(instruction, "amount_usd", 0)

        if amount <= 0:
            return TradeResult(
                success=False, from_token=from_tok, to_token=to_tok,
                amount_token=amount, error="Amount must be > 0",
            )

        if self._paper_trade:
            log.info(
                "PAPER TRADE: %s %s → %s (%.6f tokens, ~$%.2f)",
                action.upper() if action else "swap", from_tok, to_tok, amount, amount_usd,
            )
            return TradeResult(
                success=True,
                tx_hash=f"paper_{secrets.token_hex(8)}",
                from_token=from_tok, to_token=to_tok, amount_token=amount,
            )

        try:
            result = self._run_twak([
                "swap",
                "--amount", str(amount),
                "--from", from_tok,
                "--to", to_tok,
            ], timeout=SWAP_TIMEOUT_SECONDS)

            # Parse tx hash from TWAK output
            tx_hash = self._parse_tx_hash(result.stdout)
            log.info(
                "EXECUTED: %s %.6f %s → %s | tx=%s",
                action.upper(), amount, from_tok, to_tok, tx_hash,
            )
            return TradeResult(
                success=True, tx_hash=tx_hash,
                from_token=from_tok, to_token=to_tok, amount_token=amount,
            )

        except RuntimeError as exc:
            log.error("Swap execution failed: %s", exc)
            return TradeResult(
                success=False, from_token=from_tok, to_token=to_tok,
                amount_token=amount, error=str(exc),
            )

    def _parse_tx_hash(self, stdout: str) -> Optional[str]:
        """Extract transaction hash from TWAK swap output."""
        # TWAK outputs a 0x-prefixed 64-hex-char tx hash.
        # Look for a line containing it.
        for line in stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("0x") and len(stripped) >= 66:
                # Could be a tx hash: 0x + 64 hex chars
                candidate = stripped.split()[0]  # take first token
                if len(candidate) == 66 and all(c in "0123456789abcdefABCDEFx" for c in candidate):
                    return candidate
        # Fallback: return first 0x... substring
        import re
        m = re.search(r'(0x[a-fA-F0-9]{64})', stdout)
        return m.group(1) if m else None

    # ── Portfolio queries ────────────────────────────────────────────────

    def fetch_holdings(self) -> Holdings:
        """
        Fetch current wallet holdings via TWAK CLI.

        Tries --format json first for structured output. Falls back to
        plain-text table parsing. If both paths fail to identify tokens,
        raises RuntimeError rather than returning an empty Holdings —
        silent empty returns would make the orchestrator think the wallet
        is drained and trigger the 25% drawdown emergency brake.

        Returns:
            Holdings object with token balances and USD values.

        Raises:
            RuntimeError: If parsing produces zero tokens from non-empty output.
        """
        if self._paper_trade:
            log.info("PAPER TRADE: fetch_holdings — returning simulated $10,000 USDT")
            h = Holdings()
            h.tokens = {"USDT": {"balance": 10_000.0, "cost_basis_usd": 10_000.0}}
            h.total_value_usd = 10_000.0
            return h

        # Try JSON format first (TWAK may support --format json)
        raw_stdout: str = ""
        for attempt, extra_args in enumerate([
            ["--format", "json"],
            [],   # fallback: plain text
        ]):
            try:
                result = self._run_twak(
                    ["wallet", "portfolio"] + extra_args,
                    timeout=FETCH_TIMEOUT_SECONDS,
                )
                raw_stdout = result.stdout.strip()
            except RuntimeError as exc:
                log.warning("TWAK portfolio fetch attempt %d failed: %s", attempt + 1, exc)
                if attempt == 0:
                    continue  # try fallback
                raise  # both attempts failed

            holdings = self._parse_holdings(raw_stdout)
            if holdings.tokens or holdings.total_value_usd > 0:
                return holdings
            # Empty parse — try next attempt
            log.warning(
                "TWAK portfolio parse returned zero tokens (attempt %d). "
                "Raw output (first 200 chars): %.200s",
                attempt + 1, raw_stdout,
            )

        # Both attempts produced zero tokens — CRITICAL: do NOT return empty.
        # An empty Holdings would make the orchestrator think the wallet is
        # drained and trigger the 25% drawdown emergency sell.
        raise RuntimeError(
            "Failed to parse TWAK portfolio output. Both JSON and table "
            "parsing returned zero tokens from non-empty CLI output. "
            "Raw output (first 300 chars):\n" + raw_stdout[:300]
        )

    def _parse_holdings(self, stdout: str) -> Holdings:
        """Parse TWAK portfolio output into Holdings dataclass.

        Tries JSON first (TWAK --format json), then falls back to regex
        table parsing. Logs the raw output when both paths fail so the
        operator can diagnose CLI format changes.
        """
        holdings = Holdings(raw_output=stdout)

        # Try JSON first (--format json output)
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            data = None

        if isinstance(data, dict):
            tokens_list = data.get("tokens") or data.get("holdings") or []
            for entry in tokens_list:
                sym = entry.get("symbol") or entry.get("token", "?")
                holdings.tokens[sym] = {
                    "balance": float(entry.get("balance", 0)),
                    "value_usd": float(entry.get("value_usd") or entry.get("value", 0)),
                    "cost_basis_usd": float(entry.get("cost_basis_usd") or entry.get("cost_basis", 0)),
                }
            holdings.total_value_usd = float(data.get("total_value_usd") or data.get("total", 0))
            return holdings

        # Table fallback: whitespace-separated columns
        # Matches lines like "BNB   3.5   1050.00" or "CAKE   500.00000000   1200.50"
        import re
        token_pattern = re.compile(
            r'^\s*([A-Za-z0-9一-鿿]+)\s+([\d.]+)\s+([\d.]+)',
        )
        for line in stdout.splitlines():
            m = token_pattern.match(line)
            if m:
                sym = m.group(1)
                # Skip header rows that happen to match the pattern
                if sym.lower() in ("token", "symbol", "asset", "name", "---", "total"):
                    continue
                balance = float(m.group(2))
                value = float(m.group(3))
                holdings.tokens[sym] = {
                    "balance": balance,
                    "value_usd": value,
                    "cost_basis_usd": value,
                }
                holdings.total_value_usd += value

        if not holdings.tokens and stdout.strip():
            log.warning(
                "Could not parse TWAK portfolio output. No JSON and no "
                "table rows matched. Raw (first 300 chars):\n%.300s",
                stdout,
            )

        return holdings

    # ── x402 payment signing ─────────────────────────────────────────────

    def sign_x402_payment(
        self,
        challenge: dict[str, Any],
        expected_to: str,
    ) -> dict[str, Any]:
        """
        Sign an x402 payment for a CMC API call.

        Args:
            challenge: Parsed x402 challenge body. Must contain:
                - accepts[0].asset: token contract address
                - accepts[0].payTo: payee address
                - accepts[0].amount: price in base units (string or int)
                - accepts[0].maxTimeoutSeconds: validity window
                - accepts[0].extra.name: EIP-712 domain name
                - accepts[0].extra.version: EIP-712 domain version
                - accepts[0].network: "eip155:<chain_id>"
            expected_to: Payee address the caller commits to (hardcoded,
                NOT taken from the challenge body). Compared byte-equal
                against message['to'] by X402Signer.

        Returns:
            dict with:
                - envelope: base64-encoded X-PAYMENT header value
                - signature: hex signature string
                - message: the signed EIP-712 message
                - nonce: the nonce used

        Raises:
            X402SignerError: On any signing guard violation.
            ValueError: On malformed challenge.

        Reference: x402 v2 spec — the envelope is base64(json{...}).
        """
        accepts = challenge.get("accepts", [])
        if not accepts:
            raise ValueError("x402 challenge has no 'accepts' entries")

        accept = accepts[0]
        scheme = accept.get("scheme")
        if scheme != "exact":
            raise ValueError(f"Unsupported x402 scheme: {scheme!r} (expected 'exact')")

        # Parse network from "eip155:56" format
        network_str = accept.get("network", "")
        if not network_str.startswith("eip155:"):
            raise ValueError(f"Expected eip155: network, got {network_str!r}")
        chain_id = int(network_str.split(":")[1])

        asset = accept["asset"]
        pay_to = accept["payTo"]
        amount = int(accept["amount"])
        max_timeout = int(accept.get("maxTimeoutSeconds", 300))
        extra = accept.get("extra", {})

        # Build EIP-712 payload
        domain = {
            "name": extra.get("name", PAYMENT_TOKEN_EIP712_NAME),
            "version": extra.get("version", PAYMENT_TOKEN_EIP712_VERSION),
            "chainId": chain_id,
            "verifyingContract": asset,
        }
        types = {
            "EIP712Domain": EIP712_DOMAIN_FIELDS,
            "TransferWithAuthorization": TWA_FIELDS,
        }

        now = int(time.time())
        nonce = "0x" + secrets.token_hex(32)
        message = {
            "from": self._wallet.address,
            "to": pay_to,
            "value": amount,
            "validAfter": now - 60,
            "validBefore": now + max_timeout,
            "nonce": nonce,
        }

        # Sign with defense-in-depth
        try:
            signed = self._x402.sign_payment(
                domain=domain,
                types=types,
                message=message,
                expected_to=expected_to,
            )
        except (X402RecipientMismatchError, X402AmountExceededError,
                X402BudgetExhaustedError, X402PolicyError) as exc:
            log.error("x402 signing refused: %s", exc)
            raise

        # Normalize signature to 0x-prefixed hex
        raw_sig = signed["signature"]
        if hasattr(raw_sig, "hex") and not isinstance(raw_sig, str):
            sig = "0x" + raw_sig.hex()
        elif isinstance(raw_sig, str) and not raw_sig.startswith("0x"):
            sig = "0x" + raw_sig
        else:
            sig = str(raw_sig)

        # Build x402 v2 envelope
        import base64
        envelope_data = {
            "x402Version": 2,
            "scheme": scheme,
            "network": network_str,
            "payload": {
                "authorization": {
                    "from": message["from"],
                    "to": message["to"],
                    "value": str(message["value"]),
                    "validAfter": str(message["validAfter"]),
                    "validBefore": str(message["validBefore"]),
                    "nonce": message["nonce"],
                },
                "signature": sig,
            },
        }
        envelope = base64.b64encode(json.dumps(envelope_data).encode()).decode()

        log.info(
            "x402 payment signed: value=%d, to=%s, expected_to=%s, sig=%s…%s",
            amount, pay_to, expected_to, sig[:10], sig[-6:],
        )

        return {
            "envelope": envelope,
            "signature": sig,
            "message": message,
            "nonce": nonce,
        }

    # ── Full plan execution ──────────────────────────────────────────────

    def execute_plan(self, swap_plan) -> list[TradeResult]:
        """
        Execute all swaps in a SwapPlan, with delays between each.

        Each swap waits SWAP_DELAY_SECONDS after the previous one to avoid
        nonce collisions from TWAK broadcasting multiple txs to the mempool
        simultaneously. On BSC (3-second blocks), 8 seconds gives 2+ blocks
        of separation.

        Args:
            swap_plan: SwapPlan from portfolio.generate_swap_plan().

        Returns:
            List of TradeResult, one per swap attempt.
            Stops on first failure (does NOT continue after a revert).
        """
        results: list[TradeResult] = []
        for i, instruction in enumerate(swap_plan.swaps):
            if i > 0:
                log.info(
                    "Waiting %ds before next swap (nonce safety, %d/%d executed)",
                    SWAP_DELAY_SECONDS, i, len(swap_plan.swaps),
                )
                time.sleep(SWAP_DELAY_SECONDS)

            result = self.execute_swap(instruction)
            results.append(result)
            if not result.success:
                log.warning(
                    "Swap %d/%d failed, stopping plan execution: %s",
                    i + 1, len(swap_plan.swaps), result.error,
                )
                break

        return results
