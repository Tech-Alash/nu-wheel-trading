"""
Wheel Executor — executes Wheel strategy trades through IBKR.

Safety levels:
  CONFIRM  (default) — show order, wait for your "y" before sending
  AUTO              — send automatically (enable after 2+ weeks of testing)
"""

import datetime as dt
import logging

from src.ibkr.connector import IBKRConnector
from src.ibkr.data_collector import IBKRDataCollector
from src.strategies.portfolio_tracker import (
    record_sell_put,
    record_put_expired_otm,
    record_assignment,
    record_sell_call,
    record_called_away,
)
from src.models.strike_optimizer import optimize_put_strike, optimize_call_strike

logger = logging.getLogger(__name__)

# Change to "AUTO" only after system is validated
EXECUTION_MODE = "CONFIRM"


class WheelExecutor:
    """
    Executes and tracks Wheel strategy trades via IBKR API.

    Typical daily flow:
        executor = WheelExecutor(ibkr)
        executor.execute_put_signal(signal)      # From signal_generator
        executor.check_and_close_profits()        # Take-profit sweep
        executor.sync_portfolio_from_ibkr()       # Sync local state
    """

    def __init__(self, ibkr: IBKRConnector):
        self.ibkr = ibkr
        self.collector = IBKRDataCollector(ibkr)

    # ── Selling puts ────────────────────────────────────────────────────────
    def execute_put_signal(self, signal: dict) -> bool:
        """
        Execute a cash-secured put signal.

        signal keys: ticker, best_strike, best_expiry, best_premium, contracts
        """
        ticker    = signal["ticker"]
        strike    = signal["best_strike"]
        expiry    = signal["best_expiry"]          # YYYY-MM-DD
        contracts = signal.get("contracts", 1)

        # Get real-time price from IBKR
        exp_fmt = expiry.replace("-", "")
        live = self.ibkr.get_option_price(ticker, exp_fmt, strike, "P")
        if live:
            bid = live.get("bid", 0) or 0
            ask = live.get("ask", 0) or 0
            mid = (bid + ask) / 2 if bid and ask else signal.get("best_premium", 0)
            live_premium = mid
        else:
            live_premium = signal.get("best_premium", 0)

        # Limit price: mid minus small buffer for fills
        limit_price = round(live_premium * 0.98, 2)
        limit_price = max(limit_price, 0.05)

        capital_required = strike * 100 * contracts

        print(f"\n{'='*55}")
        print(f"  SELL PUT SIGNAL")
        print(f"  Ticker    : {ticker}")
        print(f"  Strike    : ${strike}")
        print(f"  Expiry    : {expiry}")
        print(f"  Contracts : {contracts}")
        print(f"  Live bid  : ${bid:.2f}  ask: ${ask:.2f}  mid: ${live_premium:.2f}")
        print(f"  Limit     : ${limit_price:.2f}")
        print(f"  Capital   : ${capital_required:,.0f}")
        print(f"  Score     : {signal.get('composite_score', '-')}")
        print(f"{'='*55}")

        if not self._confirm("Execute this trade?"):
            print("  Skipped.")
            return False

        order_id = self.ibkr.sell_put(ticker, exp_fmt, strike, contracts, limit_price)
        print(f"  Order placed — ID: {order_id}")

        # Wait for fill
        print("  Waiting for fill...")
        status = self.ibkr.wait_for_fill(order_id, timeout=30)
        fill_price = status.get("avg_fill_price", limit_price)

        if status.get("status") == "Filled":
            print(f"  FILLED @ ${fill_price:.2f}")
            record_sell_put(ticker, strike, expiry, fill_price, contracts,
                            notes=f"orderId={order_id}")
            return True
        else:
            print(f"  Not filled — status: {status.get('status', 'unknown')}")
            return False

    # ── Selling calls ───────────────────────────────────────────────────────
    def execute_call_signal(self, signal: dict) -> bool:
        """Execute a covered call signal (after assignment)."""
        ticker    = signal["ticker"]
        strike    = signal["strike"]
        expiry_dt = signal.get("expiry", "")
        contracts = signal.get("contracts", 1)

        # Calculate expiry if not provided (30 DTE from today)
        if not expiry_dt:
            exp_date = dt.date.today() + dt.timedelta(days=30)
            expiry_dt = exp_date.isoformat()

        exp_fmt = expiry_dt.replace("-", "")

        # Get live call price
        live = self.ibkr.get_option_price(ticker, exp_fmt, strike, "C")
        if live:
            bid = live.get("bid", 0) or 0
            ask = live.get("ask", 0) or 0
            live_premium = (bid + ask) / 2 if bid and ask else signal.get("premium", 0)
        else:
            live_premium = signal.get("premium", 0)

        limit_price = round(live_premium * 0.98, 2)
        limit_price = max(limit_price, 0.05)

        print(f"\n{'='*55}")
        print(f"  SELL COVERED CALL")
        print(f"  Ticker   : {ticker}")
        print(f"  Strike   : ${strike}")
        print(f"  Expiry   : {expiry_dt}")
        print(f"  Premium  : ${live_premium:.2f}  Limit: ${limit_price:.2f}")
        print(f"  Above CB : {signal.get('above_cost_basis', '?')}")
        print(f"{'='*55}")

        if not self._confirm("Execute this trade?"):
            print("  Skipped.")
            return False

        order_id = self.ibkr.sell_call(ticker, exp_fmt, strike, contracts, limit_price)
        status = self.ibkr.wait_for_fill(order_id, timeout=30)
        fill_price = status.get("avg_fill_price", limit_price)

        if status.get("status") == "Filled":
            print(f"  FILLED @ ${fill_price:.2f}")
            record_sell_call(ticker, strike, expiry_dt, fill_price, contracts)
            return True
        else:
            print(f"  Not filled — status: {status.get('status', 'unknown')}")
            return False

    # ── Take-profit sweep ───────────────────────────────────────────────────
    def check_and_close_profits(
        self,
        positions: list,
        take_profit_pct: float = 0.50,
    ):
        """
        Sweep active positions and close those that have hit take-profit target.

        Default: close when 50% of max premium is captured.
        """
        from src.db.database import load_portfolio
        portfolio = load_portfolio()

        for pos in portfolio.positions:
            if pos.phase not in ("cash_secured_put", "covered_call"):
                continue

            exp_fmt = pos.expiration.replace("-", "")
            right = "P" if pos.phase == "cash_secured_put" else "C"

            live = self.ibkr.get_option_price(pos.ticker, exp_fmt, pos.strike, right)
            if not live:
                continue

            bid = live.get("bid", 0) or 0
            ask = live.get("ask", 0) or 0
            current_price = (bid + ask) / 2

            original_premium = pos.premium_collected / (pos.contracts * 100)
            if original_premium <= 0:
                continue

            profit_pct = (original_premium - current_price) / original_premium * 100

            exp_date = dt.datetime.strptime(pos.expiration, "%Y-%m-%d").date()
            dte = (exp_date - dt.date.today()).days

            logger.info(
                f"{pos.ticker} {right}: profit={profit_pct:.0f}% | "
                f"DTE={dte} | current=${current_price:.2f}"
            )

            # Close if we've captured take_profit_pct of the premium
            if profit_pct >= take_profit_pct * 100 and dte > 2:
                buyback_price = round(current_price * 1.02, 2)
                print(f"\n  TAKE PROFIT: {pos.ticker} {right} {profit_pct:.0f}% captured")
                print(f"  Buy back @ ${buyback_price:.2f}")

                if self._confirm("Close position?"):
                    order_id = self.ibkr.buy_to_close(
                        pos.ticker, exp_fmt, pos.strike, right,
                        pos.contracts, buyback_price
                    )
                    status = self.ibkr.wait_for_fill(order_id, timeout=30)
                    if status.get("status") == "Filled":
                        print(f"  Closed @ ${status['avg_fill_price']:.2f}")
                        if right == "P":
                            record_put_expired_otm(pos.ticker)
                    else:
                        print(f"  Close order not filled: {status.get('status')}")

    # ── Portfolio sync ──────────────────────────────────────────────────────
    def sync_portfolio_from_ibkr(self):
        """
        Sync local portfolio state with actual IBKR positions.
        Useful for catching assignments and expirations automatically.
        """
        account = self.ibkr.get_account_summary()
        ibkr_positions = self.ibkr.get_positions()

        print(f"\n  IBKR Account Summary:")
        print(f"  Net Liquidation : ${float(account.get('NetLiquidation', 0)):>15,.2f}")
        print(f"  Available Funds : ${float(account.get('AvailableFunds', 0)):>15,.2f}")
        print(f"  Total Cash      : ${float(account.get('TotalCashValue', 0)):>15,.2f}")
        if ibkr_positions:
            print(f"\n  Open IBKR Positions:")
            for ticker, pos in ibkr_positions.items():
                print(f"    {ticker:8s} {pos['sec_type']:5s} qty={pos['position']:>6} @ ${pos['avg_cost']:.2f}")
        else:
            print("  No open positions in IBKR")

        return account, ibkr_positions

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _confirm(message: str) -> bool:
        """Ask for confirmation unless in AUTO mode."""
        if EXECUTION_MODE == "AUTO":
            return True
        answer = input(f"\n  {message} [y/N]: ").strip().lower()
        return answer == "y"
