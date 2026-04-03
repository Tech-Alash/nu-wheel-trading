"""
Daily runner - execute this each trading day to get signals.

Usage:
    python run_daily.py                  # Full screen + signals
    python run_daily.py --report         # Portfolio report only
    python run_daily.py --backtest AAPL  # Backtest a ticker
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.strategies.signal_generator import generate_signals, print_signals
from src.strategies.portfolio_tracker import portfolio_report
from src.backtest.backtester import backtest_wheel, backtest_portfolio


def main():
    parser = argparse.ArgumentParser(description="Wheel Strategy Daily Runner")
    parser.add_argument("--report", action="store_true", help="Show portfolio report")
    parser.add_argument("--backtest", nargs="+", help="Backtest tickers (e.g., AAPL MSFT)")
    parser.add_argument("--start", default="2024-01-01", help="Backtest start date")
    parser.add_argument("--end", default="2025-12-31", help="Backtest end date")
    parser.add_argument("--top", type=int, default=5, help="Number of top signals")

    args = parser.parse_args()

    if args.report:
        portfolio_report()
    elif args.backtest:
        if len(args.backtest) == 1:
            print(f"\nBacktesting Wheel strategy on {args.backtest[0]}...")
            result = backtest_wheel(args.backtest[0], args.start, args.end)
            print("\n=== BACKTEST RESULTS ===")
            for k, v in result.summary().items():
                print(f"  {k}: {v}")
        else:
            print(f"\nBacktesting portfolio: {', '.join(args.backtest)}...")
            results = backtest_portfolio(args.backtest, args.start, args.end)
            print("\n=== PORTFOLIO BACKTEST ===")
            print("\nPer Ticker:")
            for ticker, summary in results.get("per_ticker", {}).items():
                print(f"\n  {ticker}:")
                for k, v in summary.items():
                    print(f"    {k}: {v}")
            print("\nAggregate:")
            for k, v in results.get("aggregate", {}).items():
                print(f"  {k}: {v}")
    else:
        signals = generate_signals(top_n=args.top)
        print_signals(signals)


if __name__ == "__main__":
    main()
