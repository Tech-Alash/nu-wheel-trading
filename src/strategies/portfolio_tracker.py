"""Portfolio tracker — log trades, track P&L, generate reports.

Use this after executing trades in IBKR to keep the system in sync.
All mutations run inside SQLite transactions — no partial writes possible.
"""

import datetime as dt
import logging

import pandas as pd

from src.config import SHARES_PER_CONTRACT, STARTING_CAPITAL
from src.strategies.risk_manager import Portfolio, Position
from src.db.database import (
    load_portfolio,
    save_portfolio,
    record_trade,
    get_trade_history,
    get_closed_positions,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trade recording helpers
# ---------------------------------------------------------------------------

def record_sell_put(
    ticker: str,
    strike: float,
    expiration: str,
    premium_per_contract: float,
    contracts: int = 1,
    notes: str = "",
) -> Portfolio:
    """Record opening a new cash-secured put position."""
    portfolio = load_portfolio()

    position = Position(
        ticker=ticker,
        phase="cash_secured_put",
        entry_date=dt.date.today().isoformat(),
        strike=strike,
        expiration=expiration,
        premium_collected=premium_per_contract * SHARES_PER_CONTRACT * contracts,
        contracts=contracts,
        notes=notes,
    )
    portfolio.positions.append(position)
    portfolio.total_premium_collected += position.premium_collected

    # Single atomic write: portfolio state + trade log in one transaction block
    save_portfolio(portfolio)
    record_trade(
        action="SELL_PUT",
        ticker=ticker,
        strike=strike,
        expiration=expiration,
        premium=premium_per_contract,
        contracts=contracts,
        total_premium=position.premium_collected,
        notes=notes,
    )

    print(
        f"Recorded: SELL {contracts}x {ticker} ${strike}P exp {expiration} "
        f"@ ${premium_per_contract:.2f} (total: ${position.premium_collected:.2f})"
    )
    return portfolio


def record_put_expired_otm(ticker: str) -> Portfolio | None:
    """Record that a put expired worthless (OTM)."""
    portfolio = load_portfolio()

    position = next(
        (p for p in portfolio.positions if p.ticker == ticker and p.phase == "cash_secured_put"),
        None,
    )
    if position is None:
        print(f"No active CSP found for {ticker}")
        return None

    portfolio.total_realized_pnl += position.premium_collected
    portfolio.positions.remove(position)

    save_portfolio(portfolio)
    record_trade(
        action="PUT_EXPIRED_OTM",
        ticker=ticker,
        strike=position.strike,
        total_premium=position.premium_collected,
        pnl=position.premium_collected,
    )

    print(
        f"Recorded: {ticker} ${position.strike}P expired OTM. "
        f"Kept ${position.premium_collected:.2f} premium."
    )
    return portfolio


def record_assignment(ticker: str) -> Portfolio | None:
    """Record that put was assigned (we now own 100 shares per contract)."""
    portfolio = load_portfolio()

    position = next(
        (p for p in portfolio.positions if p.ticker == ticker and p.phase == "cash_secured_put"),
        None,
    )
    if position is None:
        print(f"No active CSP found for {ticker}")
        return None

    position.phase = "assigned"
    position.shares = SHARES_PER_CONTRACT * position.contracts
    position.cost_basis = position.strike - (position.premium_collected / position.shares)

    assignment_cost = position.strike * position.shares
    portfolio.cash -= assignment_cost

    save_portfolio(portfolio)
    record_trade(
        action="ASSIGNED",
        ticker=ticker,
        strike=position.strike,
        contracts=position.contracts,
        notes=f"shares={position.shares} cost_basis={position.cost_basis:.2f}",
    )

    print(
        f"Recorded: Assigned {position.shares} shares of {ticker} "
        f"@ ${position.strike:.2f}. Effective cost basis: ${position.cost_basis:.2f}"
    )
    return portfolio


def record_sell_call(
    ticker: str,
    strike: float,
    expiration: str,
    premium_per_contract: float,
    contracts: int = 1,
) -> Portfolio | None:
    """Record selling a covered call on assigned shares."""
    portfolio = load_portfolio()

    position = next(
        (p for p in portfolio.positions if p.ticker == ticker and p.phase == "assigned"),
        None,
    )
    if position is None:
        print(f"No assigned shares found for {ticker}")
        return None

    position.phase = "covered_call"
    position.strike = strike
    position.expiration = expiration
    cc_premium = premium_per_contract * SHARES_PER_CONTRACT * contracts
    position.premium_collected += cc_premium
    portfolio.total_premium_collected += cc_premium
    portfolio.cash += cc_premium  # Receive premium upfront

    save_portfolio(portfolio)
    record_trade(
        action="SELL_CALL",
        ticker=ticker,
        strike=strike,
        expiration=expiration,
        premium=premium_per_contract,
        contracts=contracts,
        total_premium=cc_premium,
    )

    print(
        f"Recorded: SELL {contracts}x {ticker} ${strike}C exp {expiration} "
        f"@ ${premium_per_contract:.2f}"
    )
    return portfolio


def record_call_expired_otm(ticker: str) -> Portfolio | None:
    """Call expired OTM — still hold shares, can sell another call."""
    portfolio = load_portfolio()

    position = next(
        (p for p in portfolio.positions if p.ticker == ticker and p.phase == "covered_call"),
        None,
    )
    if position is None:
        print(f"No active covered call found for {ticker}")
        return None

    position.phase = "assigned"  # Back to holding shares

    save_portfolio(portfolio)
    record_trade(action="CALL_EXPIRED_OTM", ticker=ticker, strike=position.strike)

    print(f"Recorded: {ticker} covered call expired OTM. Still holding {position.shares} shares.")
    return portfolio


def record_called_away(ticker: str) -> Portfolio | None:
    """Shares called away — position complete, back to cash."""
    portfolio = load_portfolio()

    position = next(
        (p for p in portfolio.positions if p.ticker == ticker and p.phase == "covered_call"),
        None,
    )
    if position is None:
        print(f"No active covered call found for {ticker}")
        return None

    sale_proceeds = position.strike * position.shares
    stock_pnl = (position.strike - position.cost_basis) * position.shares
    total_pnl = position.premium_collected + stock_pnl

    portfolio.cash += sale_proceeds
    portfolio.total_realized_pnl += total_pnl
    portfolio.positions.remove(position)

    save_portfolio(portfolio)
    record_trade(
        action="CALLED_AWAY",
        ticker=ticker,
        strike=position.strike,
        total_premium=position.premium_collected,
        pnl=total_pnl,
        notes=f"shares={position.shares} stock_pnl={stock_pnl:.2f}",
    )

    print(
        f"Recorded: {ticker} shares called away @ ${position.strike:.2f}. "
        f"Total PnL: ${total_pnl:.2f}"
    )
    return portfolio


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def portfolio_report() -> None:
    """Print a comprehensive portfolio report."""
    portfolio = load_portfolio()

    print("\n" + "=" * 60)
    print("  PORTFOLIO REPORT")
    print("=" * 60)
    print(f"  Cash: ${portfolio.cash:,.2f}")
    print(f"  Available: ${portfolio.available_cash:,.2f}")
    print(f"  Total Premium Collected: ${portfolio.total_premium_collected:,.2f}")
    print(f"  Realized P&L: ${portfolio.total_realized_pnl:,.2f}")
    print(f"  Return: {portfolio.total_realized_pnl / STARTING_CAPITAL * 100:.2f}%")

    if portfolio.positions:
        print(f"\n  Active Positions ({len(portfolio.positions)}):")
        print("  " + "-" * 56)
        for p in portfolio.positions:
            if p.phase == "cash_secured_put":
                print(
                    f"  [{p.phase.upper()}] {p.ticker} ${p.strike}P exp {p.expiration} "
                    f"| Premium: ${p.premium_collected:.2f}"
                )
            elif p.phase in ("assigned", "covered_call"):
                print(
                    f"  [{p.phase.upper()}] {p.ticker} {p.shares} shares "
                    f"@ ${p.cost_basis:.2f} | Premium: ${p.premium_collected:.2f}"
                )
                if p.phase == "covered_call":
                    print(f"    Call: ${p.strike}C exp {p.expiration}")

    print("\n" + "=" * 60)


def trade_history_report(ticker: str = None, limit: int = 20) -> None:
    """Print recent trade history from the immutable trade_log."""
    trades = get_trade_history(ticker=ticker, limit=limit)

    print("\n" + "=" * 60)
    title = f"  TRADE HISTORY — {ticker}" if ticker else "  TRADE HISTORY (ALL)"
    print(title)
    print("=" * 60)

    if not trades:
        print("  No trades recorded yet.")
    else:
        for t in trades:
            pnl_str = f"  PnL=${t['pnl']:.2f}" if t["pnl"] is not None else ""
            print(
                f"  [{t['timestamp'][:10]}] {t['action']:20s} {t['ticker']:6s}"
                f"  strike={t['strike']}  premium={t['premium']}{pnl_str}"
            )

    print("=" * 60)


def closed_positions_report(ticker: str = None) -> None:
    """Print historically closed positions (full audit trail from SQLite)."""
    rows = get_closed_positions(ticker=ticker)
    df = pd.DataFrame(rows) if rows else pd.DataFrame()

    print("\n" + "=" * 60)
    print("  CLOSED POSITIONS (HISTORY)")
    print("=" * 60)

    if df.empty:
        print("  No closed positions yet.")
    else:
        display_cols = ["ticker", "phase", "entry_date", "strike", "expiration",
                        "premium_collected", "contracts", "updated_at"]
        print(df[display_cols].to_string(index=False))

    print("=" * 60)


if __name__ == "__main__":
    portfolio_report()
