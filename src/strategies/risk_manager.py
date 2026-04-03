"""Risk management and position sizing for the Wheel strategy competition."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    STARTING_CAPITAL,
    SHARES_PER_CONTRACT,
    MAX_POSITION_PCT,
    MIN_CASH_RESERVE_PCT,
    MAX_CONCURRENT_WHEELS,
    MAX_LOSS_PER_POSITION_PCT,
)

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a single Wheel position."""
    ticker: str
    phase: str              # "cash_secured_put", "assigned", "covered_call"
    entry_date: str
    strike: float
    expiration: str
    premium_collected: float
    contracts: int = 1
    cost_basis: float = 0.0  # Only relevant when assigned
    shares: int = 0          # Only relevant when assigned
    notes: str = ""


@dataclass
class Portfolio:
    """Portfolio state tracker."""
    cash: float = STARTING_CAPITAL
    positions: list = field(default_factory=list)
    closed_trades: list = field(default_factory=list)
    total_premium_collected: float = 0.0
    total_realized_pnl: float = 0.0

    @property
    def active_tickers(self) -> list[str]:
        return [p.ticker for p in self.positions]

    @property
    def num_positions(self) -> int:
        return len(self.positions)

    @property
    def cash_allocated(self) -> float:
        """Cash tied up in cash-secured puts."""
        total = 0
        for p in self.positions:
            if p.phase == "cash_secured_put":
                total += p.strike * SHARES_PER_CONTRACT * p.contracts
            elif p.phase in ("assigned", "covered_call"):
                total += p.cost_basis * p.shares
        return total

    @property
    def available_cash(self) -> float:
        return self.cash - self.cash_allocated

    @property
    def portfolio_value(self) -> float:
        """Approximate portfolio value (cash + position values)."""
        return self.cash + self.total_premium_collected

    def to_dict(self) -> dict:
        return {
            "cash": round(self.cash, 2),
            "available_cash": round(self.available_cash, 2),
            "num_positions": self.num_positions,
            "active_tickers": self.active_tickers,
            "total_premium_collected": round(self.total_premium_collected, 2),
            "total_realized_pnl": round(self.total_realized_pnl, 2),
            "cash_allocated": round(self.cash_allocated, 2),
        }


class RiskManager:
    """Manages risk constraints for the competition."""

    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio

    def can_open_new_wheel(self, ticker: str, strike: float, contracts: int = 1) -> tuple[bool, str]:
        """Check if we can open a new cash-secured put position."""

        # Rule: no double entries on same ticker
        if ticker in self.portfolio.active_tickers:
            return False, f"{ticker} already has an active wheel"

        # Max concurrent wheels
        if self.portfolio.num_positions >= MAX_CONCURRENT_WHEELS:
            return False, f"Max {MAX_CONCURRENT_WHEELS} concurrent wheels reached"

        # Position size check
        position_cost = strike * SHARES_PER_CONTRACT * contracts
        max_allowed = self.portfolio.cash * MAX_POSITION_PCT
        if position_cost > max_allowed:
            return False, f"Position ${position_cost:,.0f} exceeds max ${max_allowed:,.0f} ({MAX_POSITION_PCT*100}%)"

        # Cash reserve check
        min_reserve = self.portfolio.cash * MIN_CASH_RESERVE_PCT
        if self.portfolio.available_cash - position_cost < min_reserve:
            return False, f"Would leave less than {MIN_CASH_RESERVE_PCT*100}% cash reserve"

        # Available cash check
        if position_cost > self.portfolio.available_cash:
            return False, f"Insufficient cash: need ${position_cost:,.0f}, have ${self.portfolio.available_cash:,.0f}"

        return True, "OK"

    def calculate_position_size(
        self, strike: float, portfolio_value: float = None
    ) -> int:
        """Calculate how many contracts to sell (usually 1 for this competition)."""
        if portfolio_value is None:
            portfolio_value = self.portfolio.cash

        max_position = portfolio_value * MAX_POSITION_PCT
        cost_per_contract = strike * SHARES_PER_CONTRACT
        max_contracts = int(max_position // cost_per_contract)

        return max(1, min(max_contracts, 3))  # Cap at 3 contracts per ticker

    def calculate_max_loss(self, strike: float, contracts: int = 1) -> float:
        """Theoretical max loss for a cash-secured put (stock goes to 0)."""
        return strike * SHARES_PER_CONTRACT * contracts

    def sector_exposure(self) -> dict:
        """Check sector concentration."""
        sectors = {}
        for p in self.portfolio.positions:
            sector = getattr(p, "sector", "Unknown")
            sectors[sector] = sectors.get(sector, 0) + 1
        return sectors

    def daily_risk_report(self) -> dict:
        """Generate a daily risk assessment."""
        return {
            "portfolio_summary": self.portfolio.to_dict(),
            "utilization_pct": round(
                self.portfolio.cash_allocated / self.portfolio.cash * 100, 2
            ) if self.portfolio.cash > 0 else 0,
            "available_for_new_positions": round(self.portfolio.available_cash, 2),
            "max_new_position_size": round(
                self.portfolio.cash * MAX_POSITION_PCT, 2
            ),
            "positions_remaining": MAX_CONCURRENT_WHEELS - self.portfolio.num_positions,
        }

    def should_close_early(
        self,
        position: Position,
        current_option_price: float,
        days_to_expiry: int,
    ) -> tuple[bool, str]:
        """Determine if a position should be closed early.

        Close early if:
        - 80%+ of max profit captured with >50% time remaining
        - Position is at risk of max loss
        """
        if position.phase == "cash_secured_put":
            profit_pct = (position.premium_collected - current_option_price) / position.premium_collected * 100

            # Close at 50% profit if more than half the time remains
            if profit_pct >= 50 and days_to_expiry > 7:
                return True, f"Take profit: {profit_pct:.0f}% of max profit captured"

            # Close at 80% profit regardless
            if profit_pct >= 80:
                return True, f"Take profit: {profit_pct:.0f}% of max profit captured"

        return False, "Hold"


def allocate_capital(
    portfolio: Portfolio,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Decide how to allocate capital across top candidates.

    Returns candidates with added columns: contracts, capital_allocated, approved.
    """
    rm = RiskManager(portfolio)
    results = []

    for _, row in candidates.iterrows():
        ticker = row["ticker"]
        strike = row.get("best_strike", 0)

        if strike <= 0:
            continue

        can_trade, reason = rm.can_open_new_wheel(ticker, strike)
        contracts = rm.calculate_position_size(strike) if can_trade else 0

        results.append({
            **row.to_dict(),
            "contracts": contracts,
            "capital_required": strike * SHARES_PER_CONTRACT * contracts,
            "approved": can_trade,
            "reason": reason,
        })

    return pd.DataFrame(results)
