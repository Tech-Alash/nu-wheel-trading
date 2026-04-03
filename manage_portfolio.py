"""
Interactive portfolio management - record trades after executing in IBKR.

Usage:
    python manage_portfolio.py sell_put AAPL 170 2026-05-16 2.50
    python manage_portfolio.py expired AAPL
    python manage_portfolio.py assigned AAPL
    python manage_portfolio.py sell_call AAPL 180 2026-05-16 1.80
    python manage_portfolio.py call_expired AAPL
    python manage_portfolio.py called_away AAPL
    python manage_portfolio.py report
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.strategies.portfolio_tracker import (
    record_sell_put,
    record_put_expired_otm,
    record_assignment,
    record_sell_call,
    record_call_expired_otm,
    record_called_away,
    portfolio_report,
)


def main():
    parser = argparse.ArgumentParser(description="Portfolio Management")
    subparsers = parser.add_subparsers(dest="command")

    # Sell put
    sp = subparsers.add_parser("sell_put")
    sp.add_argument("ticker")
    sp.add_argument("strike", type=float)
    sp.add_argument("expiration", help="YYYY-MM-DD")
    sp.add_argument("premium", type=float, help="Premium per contract")
    sp.add_argument("--contracts", type=int, default=1)

    # Put expired OTM
    sp = subparsers.add_parser("expired")
    sp.add_argument("ticker")

    # Assigned
    sp = subparsers.add_parser("assigned")
    sp.add_argument("ticker")

    # Sell call
    sp = subparsers.add_parser("sell_call")
    sp.add_argument("ticker")
    sp.add_argument("strike", type=float)
    sp.add_argument("expiration", help="YYYY-MM-DD")
    sp.add_argument("premium", type=float, help="Premium per contract")
    sp.add_argument("--contracts", type=int, default=1)

    # Call expired OTM
    sp = subparsers.add_parser("call_expired")
    sp.add_argument("ticker")

    # Called away
    sp = subparsers.add_parser("called_away")
    sp.add_argument("ticker")

    # Report
    subparsers.add_parser("report")

    args = parser.parse_args()

    if args.command == "sell_put":
        record_sell_put(args.ticker, args.strike, args.expiration, args.premium, args.contracts)
    elif args.command == "expired":
        record_put_expired_otm(args.ticker)
    elif args.command == "assigned":
        record_assignment(args.ticker)
    elif args.command == "sell_call":
        record_sell_call(args.ticker, args.strike, args.expiration, args.premium, args.contracts)
    elif args.command == "call_expired":
        record_call_expired_otm(args.ticker)
    elif args.command == "called_away":
        record_called_away(args.ticker)
    elif args.command == "report":
        portfolio_report()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
