"""
Live runner — uses real IBKR data and optionally executes trades.

Usage:
    python run_live.py                    # Signals + manual confirmation
    python run_live.py --sync             # Sync portfolio from IBKR only
    python run_live.py --collect          # Collect daily ML data only
    python run_live.py --take-profit      # Check + close profitable positions
    python run_live.py --full             # All of the above + new signals
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from src.ibkr.connector import IBKRConnector
from src.ibkr.data_collector import IBKRDataCollector
from src.ibkr.wheel_executor import WheelExecutor
from src.strategies.signal_generator import generate_signals, print_signals, load_portfolio
from src.strategies.portfolio_tracker import portfolio_report
from src.data.fetcher import WHEEL_UNIVERSE


def main():
    parser = argparse.ArgumentParser(description="Live Wheel Strategy Runner (IBKR)")
    parser.add_argument("--sync",         action="store_true", help="Sync portfolio from IBKR")
    parser.add_argument("--collect",      action="store_true", help="Collect daily ML snapshot data")
    parser.add_argument("--take-profit",  action="store_true", help="Check and close profitable positions")
    parser.add_argument("--full",         action="store_true", help="Run everything")
    parser.add_argument("--top",          type=int, default=5, help="Number of signals")
    parser.add_argument("--client-id",    type=int, default=10, help="IBKR client ID (must be unique)")
    args = parser.parse_args()

    # Connect to IB Gateway
    print("\nConnecting to IB Gateway (port 4002)...")
    ibkr = IBKRConnector(port=4002, client_id=args.client_id)
    if not ibkr.connect(timeout=10):
        print("ERROR: Could not connect to IB Gateway.")
        print("Make sure IB Gateway is running and API is enabled on port 4002.")
        sys.exit(1)

    print(f"Connected — Account: {ibkr.app.account_id}")

    collector = IBKRDataCollector(ibkr)
    executor  = WheelExecutor(ibkr)
    portfolio = load_portfolio()

    try:
        # 1. Portfolio sync
        if args.sync or args.full:
            print("\n--- Syncing portfolio from IBKR ---")
            executor.sync_portfolio_from_ibkr()

        # 2. Collect daily ML snapshot data
        if args.collect or args.full:
            print("\n--- Collecting daily ML data ---")
            result = collector.collect_daily_snapshot(
                tickers=WHEEL_UNIVERSE[:20],   # Top 20 for speed
                active_positions=portfolio.positions,
            )
            print(f"  Collected: {result['stock_snapshots']} stocks, "
                  f"{result['option_snapshots']} options")

        # 3. Take-profit sweep
        if args.take_profit or args.full:
            print("\n--- Checking take-profit opportunities ---")
            executor.check_and_close_profits(portfolio.positions, take_profit_pct=0.50)

        # 4. Generate new signals
        print("\n--- Generating trade signals ---")
        signals = generate_signals(portfolio=portfolio, top_n=args.top)
        print_signals(signals)

        # 5. Execute approved signals
        if signals["new_puts"]:
            print("\n--- Execute new cash-secured puts? ---")
            for signal in signals["new_puts"]:
                executor.execute_put_signal(signal)

        if signals["new_calls"]:
            print("\n--- Execute new covered calls? ---")
            for signal in signals["new_calls"]:
                executor.execute_call_signal(signal)

        # 6. Final portfolio report
        portfolio_report()

    finally:
        ibkr.disconnect()
        print("\nDisconnected from IB Gateway.")


if __name__ == "__main__":
    main()
