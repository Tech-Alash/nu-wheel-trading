"""Daily Signal Generator - Main entry point for trade recommendations.

Run this daily to get actionable trade signals for IBKR paper trading.
Outputs: which puts to sell, which calls to sell, which positions to manage.
"""

import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import (
    STARTING_CAPITAL,
    DEFAULT_DTE_MIN,
    DEFAULT_DTE_MAX,
    LOGS_DIR,
)
from src.data.fetcher import (
    fetch_stock_data,
    compute_iv_rank,
    WHEEL_UNIVERSE,
)
from src.strategies.screener import screen_universe
from src.strategies.risk_manager import Portfolio, RiskManager, Position, allocate_capital
from src.models.strike_optimizer import optimize_put_strike, optimize_call_strike
from src.ml.predictor import get_predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / f"signals_{dt.date.today()}.log"),
    ],
)
logger = logging.getLogger(__name__)


from src.db.database import (
    load_portfolio,   # noqa: F401  (re-exported for backwards compat)
    save_portfolio,   # noqa: F401
    init_db,
)

# Initialise DB on module import (creates tables if they don't exist)
init_db()


def generate_signals(
    portfolio: Portfolio = None,
    tickers: list[str] = None,
    top_n: int = 5,
) -> dict:
    """Generate daily trading signals.

    Returns a dict with:
    - new_puts: Cash-secured puts to sell
    - new_calls: Covered calls to sell (for assigned positions)
    - manage: Positions to close or roll
    - risk_report: Current portfolio risk status
    """
    if portfolio is None:
        portfolio = load_portfolio()

    rm = RiskManager(portfolio)
    today = dt.date.today().isoformat()

    signals = {
        "date": today,
        "new_puts": [],
        "new_calls": [],
        "manage": [],
        "risk_report": rm.daily_risk_report(),
    }

    # --- 1. Screen for new put opportunities ---
    logger.info("=== Screening for new CSP opportunities ===")

    # Exclude tickers we already have positions in
    available_tickers = [t for t in (tickers or WHEEL_UNIVERSE)
                         if t not in portfolio.active_tickers]

    if available_tickers and rm.daily_risk_report()["positions_remaining"] > 0:
        screened = screen_universe(available_tickers, top_n=top_n, portfolio_value=portfolio.cash)

        if not screened.empty:
            allocated = allocate_capital(portfolio, screened)
            approved = allocated[allocated["approved"]]

            for _, row in approved.iterrows():
                signal = {
                    "action": "SELL PUT (Cash-Secured)",
                    "ticker": row["ticker"],
                    "strike": row["best_strike"],
                    "expiration": row["best_expiry"],
                    "dte": row.get("best_dte", 0),
                    "premium": row["best_premium"],
                    "contracts": row["contracts"],
                    "capital_required": row["capital_required"],
                    "otm_pct": row["otm_pct"],
                    "composite_score": row["composite_score"],
                    "iv_rank": row.get("iv_rank"),
                    "sector": row.get("sector", ""),
                }
                signals["new_puts"].append(signal)
                logger.info(
                    f"  SIGNAL: Sell {row['ticker']} ${row['best_strike']}P "
                    f"exp {row['best_expiry']} @ ${row['best_premium']:.2f} "
                    f"({row['otm_pct']:.1f}% OTM, score={row['composite_score']:.1f})"
                )

    # --- 2. Check assigned positions for covered call opportunities ---
    logger.info("=== Checking for covered call opportunities ===")

    for position in portfolio.positions:
        if position.phase == "assigned":
            hist = fetch_stock_data(position.ticker, period="3mo")
            if hist is None or hist.empty:
                continue

            current_price = hist["Close"].iloc[-1]
            iv_rank = compute_iv_rank(position.ticker)
            sigma = hist["Close"].pct_change().std() * (252 ** 0.5)

            call_opt = optimize_call_strike(
                stock_price=current_price,
                cost_basis=position.cost_basis,
                sigma=sigma if sigma > 0 else 0.25,
                dte=30,
            )

            if call_opt:
                signal = {
                    "action": "SELL CALL (Covered)",
                    "ticker": position.ticker,
                    "strike": call_opt["strike"],
                    "premium": call_opt["premium"],
                    "dte": call_opt["dte"],
                    "above_cost_basis": call_opt["above_cost_basis"],
                    "cost_basis": position.cost_basis,
                    "current_price": round(current_price, 2),
                    "prob_called_away": call_opt["prob_called_away"],
                }
                signals["new_calls"].append(signal)
                logger.info(
                    f"  SIGNAL: Sell {position.ticker} ${call_opt['strike']}C "
                    f"@ ${call_opt['premium']:.2f} "
                    f"(above cost basis: {call_opt['above_cost_basis']})"
                )

    # --- 3. Position management ---
    logger.info("=== Position Management ===")

    for position in portfolio.positions:
        if position.phase == "cash_secured_put":
            exp_date = dt.datetime.strptime(position.expiration, "%Y-%m-%d").date()
            days_to_expiry = (exp_date - dt.date.today()).days

            if days_to_expiry <= 0:
                signals["manage"].append({
                    "action": "EXPIRED",
                    "ticker": position.ticker,
                    "details": "Check if assigned or expired OTM. Update portfolio state.",
                })
            elif days_to_expiry <= 5:
                signals["manage"].append({
                    "action": "MONITOR",
                    "ticker": position.ticker,
                    "details": f"Expiring in {days_to_expiry} days. Consider rolling if near the money.",
                })

    return signals


def print_signals(signals: dict):
    """Pretty-print the daily signals."""
    print("\n" + "=" * 70)
    print(f"  DAILY TRADING SIGNALS - {signals['date']}")
    print("=" * 70)

    # Risk report
    report = signals["risk_report"]
    print(f"\n  Portfolio: Cash=${report['portfolio_summary']['cash']:,.0f} "
          f"| Allocated=${report['portfolio_summary']['cash_allocated']:,.0f} "
          f"| Available=${report['available_for_new_positions']:,.0f}")
    print(f"  Positions: {report['portfolio_summary']['num_positions']}/{report['positions_remaining'] + report['portfolio_summary']['num_positions']} "
          f"| Utilization: {report['utilization_pct']}%")
    print(f"  Premium Collected: ${report['portfolio_summary']['total_premium_collected']:,.0f}")

    # New puts
    if signals["new_puts"]:
        print(f"\n  --- NEW CASH-SECURED PUTS TO SELL ({len(signals['new_puts'])}) ---")
        for s in signals["new_puts"]:
            print(f"\n  >> {s['ticker']} | Sell ${s['strike']}P exp {s['expiration']} ({s['dte']}d)")
            print(f"     Premium: ${s['premium']:.2f}/contract | {s['otm_pct']:.1f}% OTM")
            print(f"     Contracts: {s['contracts']} | Capital: ${s['capital_required']:,.0f}")
            print(f"     Score: {s['composite_score']:.1f} | IV Rank: {s['iv_rank']}")
    else:
        print("\n  --- No new put signals today ---")

    # New calls
    if signals["new_calls"]:
        print(f"\n  --- COVERED CALLS TO SELL ({len(signals['new_calls'])}) ---")
        for s in signals["new_calls"]:
            print(f"\n  >> {s['ticker']} | Sell ${s['strike']}C ({s['dte']}d)")
            print(f"     Premium: ${s['premium']:.2f} | P(called away): {s['prob_called_away']:.1f}%")
            print(f"     Cost basis: ${s['cost_basis']:.2f} | Current: ${s['current_price']:.2f}")
    else:
        print("\n  --- No covered call signals today ---")

    # Management
    if signals["manage"]:
        print(f"\n  --- POSITION MANAGEMENT ({len(signals['manage'])}) ---")
        for s in signals["manage"]:
            print(f"  [{s['action']}] {s['ticker']}: {s['details']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    signals = generate_signals()
    print_signals(signals)
