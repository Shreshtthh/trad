// SPDX-License-Identifier: MIT
pragma solidity ^0.8.21;

/// @notice Minimal ERC20 interface.
interface IERC20Minimal {
    function balanceOf(address who) external view returns (uint256);
    function transfer(address recipient, uint256 value) external returns (bool);
    function approve(address spender, uint256 value) external returns (bool);
}

/// @notice Aave V3 pool — flashLoanSimple entry point.
interface IAaveV3Pool {
    function flashLoanSimple(
        address receiver,
        address asset,
        uint256 amount,
        bytes calldata data,
        uint16 referralCode
    ) external;
}

/// @notice PancakeSwap V2 router (same ABI as Uniswap V2).
interface IPancakeRouterV2 {
    function swapExactTokensForTokens(
        uint256 amountIn,
        uint256 minAmountOut,
        address[] calldata route,
        address recipient,
        uint256 deadline
    ) external returns (uint256[] memory amounts);
}

// ────────────────────────────────────────────────────────────────
//  TokenOps — safe ERC20 helpers
// ────────────────────────────────────────────────────────────────

library TokenOps {
    error TokenCallReverted(address token);
    error TokenCallReturnedFalse(address token);

    function safeSend(IERC20Minimal token, address to, uint256 value) internal {
        _invoke(token, abi.encodeWithSelector(token.transfer.selector, to, value));
    }

    function safeApproveExact(IERC20Minimal token, address spender, uint256 value) internal {
        bytes memory payload = abi.encodeWithSelector(token.approve.selector, spender, value);
        if (!_invokeBool(token, payload)) {
            // USDT / non-standard tokens require reset-to-zero
            _invoke(token, abi.encodeWithSelector(token.approve.selector, spender, 0));
            _invoke(token, payload);
        }
    }

    function _invoke(IERC20Minimal token, bytes memory payload) private {
        (bool ok, bytes memory ret) = address(token).call(payload);
        if (!ok) revert TokenCallReverted(address(token));
        if (ret.length > 0 && !abi.decode(ret, (bool))) revert TokenCallReturnedFalse(address(token));
    }

    function _invokeBool(IERC20Minimal token, bytes memory payload) private returns (bool) {
        (bool ok, bytes memory ret) = address(token).call(payload);
        return ok && (ret.length == 0 || abi.decode(ret, (bool)));
    }
}

// ────────────────────────────────────────────────────────────────
//  HonestFlashArbBSC — BSC flash-loan arb executor
// ────────────────────────────────────────────────────────────────

contract HonestFlashArbBSC {
    using TokenOps for IERC20Minimal;

    error Unauthorized();
    error ZeroAddress();
    error ZeroAmount();
    error BadPlan();
    error BadCallback();
    error LoanAlreadyOpen();
    error NoLoanOpen();
    error RouterNotAllowed(address router);
    error TokenNotAllowed(address token);
    error GainTooSmall();
    error ContractPaused();
    error MustBePaused();
    error NativeTransfersDisabled();

    struct ArbPlan {
        address router1;
        address router2;
        address[] path1;
        address[] path2;
        uint256 amountOutMin1;
        uint256 amountOutMin2;
        uint256 minProfit;
        uint256 deadline;
    }

    // ── Immutable config ──
    address public immutable owner;
    address public immutable pool; // Aave V3 Pool on BSC

    // ── State ──
    bool public paused;
    bool public loanOpen;
    mapping(address => bool) public routerWhitelist;
    mapping(address => bool) public tokenWhitelist;

    bytes32 public activePlanHash;
    address public activeAsset;
    uint256 public activeAmount;
    uint256 public balanceBefore;

    event PauseStatusChanged(bool paused);
    event FlashRequested(address indexed asset, uint256 amount);
    event FlashCompleted(address indexed asset, uint256 amount, uint256 premium, uint256 profit);
    event TokenRecovered(address indexed token, address indexed to, uint256 amount);

    modifier onlyOwner() {
        if (msg.sender != owner) revert Unauthorized();
        _;
    }

    modifier whenRunning() {
        if (paused) revert ContractPaused();
        _;
    }

    constructor(address pool_, address[] memory routers, address[] memory tokens) {
        if (pool_ == address(0)) revert ZeroAddress();
        owner = msg.sender;
        pool = pool_;
        for (uint256 i = 0; i < routers.length; ) {
            address r = routers[i];
            if (r == address(0)) revert ZeroAddress();
            routerWhitelist[r] = true;
            unchecked { ++i; }
        }
        for (uint256 i = 0; i < tokens.length; ) {
            address t = tokens[i];
            if (t == address(0)) revert ZeroAddress();
            tokenWhitelist[t] = true;
            unchecked { ++i; }
        }
    }

    // ── Owner controls ──

    function pause() external onlyOwner {
        paused = true;
        emit PauseStatusChanged(true);
    }

    function unpause() external onlyOwner {
        paused = false;
        emit PauseStatusChanged(false);
    }

    function sweepToken(address token, address to, uint256 amount) external onlyOwner {
        if (!paused) revert MustBePaused();
        if (token == address(0) || to == address(0)) revert ZeroAddress();
        if (amount == 0) revert ZeroAmount();
        IERC20Minimal(token).safeSend(to, amount);
        emit TokenRecovered(token, to, amount);
    }

    // ── Entry point ──

    function startArbitrage(address asset, uint256 amount, ArbPlan calldata plan)
        external onlyOwner whenRunning
    {
        if (loanOpen) revert LoanAlreadyOpen();
        if (asset == address(0)) revert ZeroAddress();
        if (amount == 0) revert ZeroAmount();
        _checkPlan(asset, plan);

        uint256 startingBalance = IERC20Minimal(asset).balanceOf(address(this));
        bytes memory encodedPlan = abi.encode(plan);

        loanOpen = true;
        activePlanHash = keccak256(encodedPlan);
        activeAsset = asset;
        activeAmount = amount;
        balanceBefore = startingBalance;

        emit FlashRequested(asset, amount);

        IAaveV3Pool(pool).flashLoanSimple(address(this), asset, amount, encodedPlan, 0);

        if (loanOpen) revert BadCallback();
    }

    // ── Aave V3 callback ──

    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata data
    ) external returns (bool) {
        if (msg.sender != pool) revert BadCallback();
        if (initiator != address(this)) revert BadCallback();
        if (!loanOpen) revert NoLoanOpen();
        if (asset != activeAsset || amount != activeAmount) revert BadCallback();
        if (keccak256(data) != activePlanHash) revert BadCallback();

        ArbPlan memory plan = abi.decode(data, (ArbPlan));
        _checkPlan(asset, plan);

        uint256 currentBalance = IERC20Minimal(asset).balanceOf(address(this));
        if (currentBalance < balanceBefore + amount) revert BadCallback();

        // Leg 1: borrowed asset → bridge token
        IERC20Minimal(asset).safeApproveExact(plan.router1, amount);
        uint256[] memory swap1 = IPancakeRouterV2(plan.router1).swapExactTokensForTokens(
            amount, plan.amountOutMin1, plan.path1, address(this), plan.deadline
        );
        IERC20Minimal(asset).safeApproveExact(plan.router1, 0);

        uint256 bridgeAmount = swap1[swap1.length - 1];
        address bridgeToken = plan.path1[plan.path1.length - 1];

        // Leg 2: bridge token → original asset
        IERC20Minimal(bridgeToken).safeApproveExact(plan.router2, bridgeAmount);
        IPancakeRouterV2(plan.router2).swapExactTokensForTokens(
            bridgeAmount, plan.amountOutMin2, plan.path2, address(this), plan.deadline
        );
        IERC20Minimal(bridgeToken).safeApproveExact(plan.router2, 0);

        uint256 debt = amount + premium;
        uint256 endingBalance = IERC20Minimal(asset).balanceOf(address(this));

        if (endingBalance < balanceBefore + debt + plan.minProfit) revert GainTooSmall();

        uint256 profit = endingBalance - balanceBefore - debt;
        _resetLoanState();

        IERC20Minimal(asset).safeApproveExact(pool, debt);

        emit FlashCompleted(asset, amount, premium, profit);
        return true;
    }

    // ── Validation ──

    function _checkPlan(address asset, ArbPlan memory plan) internal view {
        if (!tokenWhitelist[asset]) revert TokenNotAllowed(asset);
        if (!routerWhitelist[plan.router1]) revert RouterNotAllowed(plan.router1);
        if (!routerWhitelist[plan.router2]) revert RouterNotAllowed(plan.router2);
        if (plan.path1.length < 2 || plan.path2.length < 2) revert BadPlan();
        if (plan.path1[0] != asset) revert BadPlan();
        if (plan.path2[plan.path2.length - 1] != asset) revert BadPlan();
        if (plan.path1[plan.path1.length - 1] != plan.path2[0]) revert BadPlan();
        if (plan.amountOutMin1 == 0 || plan.amountOutMin2 == 0 || plan.minProfit == 0) revert BadPlan();
        if (block.timestamp > plan.deadline) revert BadPlan();

        _checkPath(plan.path1);
        _checkPath(plan.path2);
    }

    function _checkPath(address[] memory path) internal view {
        for (uint256 i = 0; i < path.length; ) {
            if (!tokenWhitelist[path[i]]) revert TokenNotAllowed(path[i]);
            unchecked { ++i; }
        }
    }

    function _resetLoanState() internal {
        loanOpen = false;
        activePlanHash = bytes32(0);
        activeAsset = address(0);
        activeAmount = 0;
        balanceBefore = 0;
    }

    receive() external payable { revert NativeTransfersDisabled(); }
    fallback() external payable { revert NativeTransfersDisabled(); }
}
