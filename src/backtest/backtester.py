"""Backtester for the Wheel strategy on historical data.

Simulates the full Cash-Secured Put → Assignment → Covered Call cycle
using historical price data and Black-Scholes option pricing.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    STARTING_CAPITAL,
    SHARES_PER_CONTRACT,
    MAX_POSITION_PCT,
    DEFAULT_DTE_MIN,
    DEFAULT_DTE_MAX,
)
from src.models.strike_optimizer import (
    black_scholes_put_price,
    black_scholes_call_price,
    optimize_put_strike,
    optimize_call_strike,
)
from src.data.fetcher import fetch_stock_data

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    ticker: str
    phase: str
    entry_date: str
    exit_date: str
    strike: float
    premium: float
    contracts: int
    outcome: str  # "expired_otm", "assigned", "called_away", "closed_early"
    pnl: float
    stock_price_entry: float
    stock_price_exit: float


@dataclass
class BacktestResult:
    trades: list
    initial_capital: float
    final_capital: float
    total_premium: float
    total_pnl: float
    num_trades: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    equity_curve: list

    def summary(self) -> dict:
        return {
            "initial_capital": self.initial_capital,
            "final_capital": round(self.final_capital, 2),
            "total_return_pct": round((self.final_capital / self.initial_capital - 1) * 100, 2),
            "total_premium_collected": round(self.total_premium, 2),
            "total_pnl": round(self.total_pnl, 2),
            "num_trades": self.num_trades,
            "win_rate": round(self.win_rate * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
        }


def estimate_iv(hist: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Estimate implied volatility using realized volatility as proxy."""
    returns = np.log(hist["Close"] / hist["Close"].shift(1))
    rv = returns.rolling(lookback).std() * np.sqrt(252)
    # IV typically trades at a premium to RV
    iv_estimate = rv * 1.2
    return iv_estimate


def backtest_wheel(
    ticker: str,
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    capital: float = STARTING_CAPITAL,
    target_dte: int = 30,
    otm_pct_put: float = 0.05,
    otm_pct_call: float = 0.05,
    take_profit_pct: float = 0.50,
) -> BacktestResult:
    """Backtest the Wheel strategy on a single ticker."""

    # Fetch historical data with buffer for IV calculation
    hist = fetch_stock_data(ticker, period="5y")
    if hist is None or hist.empty:
        raise ValueError(f"No data for {ticker}")

    hist = hist.loc[start_date:end_date].copy()
    if len(hist) < 50:
        raise ValueError(f"Insufficient data for {ticker}")

    # Estimate IV series
    iv_series = estimate_iv(hist)

    cash = capital
    shares_held = 0
    cost_basis = 0.0
    trades = []
    equity_curve = []
    active_put = None
    active_call = None

    dates = hist.index.tolist()
    risk_free_rate = 0.05

    i = 0
    while i < len(dates):
        date = dates[i]
        price = hist.loc[date, "Close"]
        iv = iv_series.loc[date] if date in iv_series.index and not np.isnan(iv_series.loc[date]) else 0.25

        # Track equity
        position_value = shares_held * price if shares_held > 0 else 0
        equity = cash + position_value
        equity_curve.append({"date": date, "equity": equity})

        # Phase 1: Sell cash-secured put if no active position
        if active_put is None and active_call is None and shares_held == 0:
            strike = round(price * (1 - otm_pct_put), 2)
            position_cost = strike * SHARES_PER_CONTRACT

            if position_cost <= cash * MAX_POSITION_PCT:
                premium = black_scholes_put_price(price, strike, target_dte / 365, risk_free_rate, iv)
                premium_total = premium * SHARES_PER_CONTRACT

                active_put = {
                    "entry_date": date,
                    "strike": strike,
                    "premium": premium,
                    "premium_total": premium_total,
                    "entry_price": price,
                    "expiry_idx": min(i + target_dte, len(dates) - 1),
                    "days_held": 0,
                }
                cash += premium_total  # Collect premium upfront

        # Check put expiration / assignment
        elif active_put is not None:
            active_put["days_held"] += 1
            days_remaining = active_put["expiry_idx"] - i

            # Check take-profit (buy back at 50% of premium)
            if days_remaining > 5:
                current_put_price = black_scholes_put_price(
                    price, active_put["strike"], max(days_remaining, 1) / 365, risk_free_rate, iv
                )
                profit_pct = (active_put["premium"] - current_put_price) / active_put["premium"]

                if profit_pct >= take_profit_pct:
                    # Close early
                    buyback_cost = current_put_price * SHARES_PER_CONTRACT
                    cash -= buyback_cost
                    pnl = active_put["premium_total"] - buyback_cost

                    trades.append(BacktestTrade(
                        ticker=ticker, phase="cash_secured_put",
                        entry_date=str(active_put["entry_date"]),
                        exit_date=str(date),
                        strike=active_put["strike"],
                        premium=active_put["premium_total"],
                        contracts=1, outcome="closed_early",
                        pnl=pnl,
                        stock_price_entry=active_put["entry_price"],
                        stock_price_exit=price,
                    ))
                    active_put = None
                    i += 1
                    continue

            # At expiration
            if i >= active_put["expiry_idx"]:
                if price >= active_put["strike"]:
                    # Expired OTM - keep premium
                    trades.append(BacktestTrade(
                        ticker=ticker, phase="cash_secured_put",
                        entry_date=str(active_put["entry_date"]),
                        exit_date=str(date),
                        strike=active_put["strike"],
                        premium=active_put["premium_total"],
                        contracts=1, outcome="expired_otm",
                        pnl=active_put["premium_total"],
                        stock_price_entry=active_put["entry_price"],
                        stock_price_exit=price,
                    ))
                    active_put = None
                else:
                    # Assigned - buy shares at strike
                    shares_held = SHARES_PER_CONTRACT
                    cost_basis = active_put["strike"] - active_put["premium"]  # Effective cost basis
                    assignment_cost = active_put["strike"] * SHARES_PER_CONTRACT
                    cash -= assignment_cost

                    trades.append(BacktestTrade(
                        ticker=ticker, phase="cash_secured_put",
                        entry_date=str(active_put["entry_date"]),
                        exit_date=str(date),
                        strike=active_put["strike"],
                        premium=active_put["premium_total"],
                        contracts=1, outcome="assigned",
                        pnl=active_put["premium_total"],  # Premium still collected
                        stock_price_entry=active_put["entry_price"],
                        stock_price_exit=price,
                    ))
                    active_put = None

        # Phase 3: Sell covered call if we hold shares
        elif shares_held > 0 and active_call is None:
            call_strike = round(price * (1 + otm_pct_call), 2)
            # Try to sell above cost basis
            if call_strike < cost_basis:
                call_strike = round(cost_basis * 1.02, 2)

            call_premium = black_scholes_call_price(price, call_strike, target_dte / 365, risk_free_rate, iv)
            call_premium_total = call_premium * SHARES_PER_CONTRACT

            active_call = {
                "entry_date": date,
                "strike": call_strike,
                "premium": call_premium,
                "premium_total": call_premium_total,
                "entry_price": price,
                "expiry_idx": min(i + target_dte, len(dates) - 1),
            }
            cash += call_premium_total

        # Check call expiration
        elif active_call is not None:
            if i >= active_call["expiry_idx"]:
                if price <= active_call["strike"]:
                    # Expired OTM - keep shares and premium
                    trades.append(BacktestTrade(
                        ticker=ticker, phase="covered_call",
                        entry_date=str(active_call["entry_date"]),
                        exit_date=str(date),
                        strike=active_call["strike"],
                        premium=active_call["premium_total"],
                        contracts=1, outcome="expired_otm",
                        pnl=active_call["premium_total"],
                        stock_price_entry=active_call["entry_price"],
                        stock_price_exit=price,
                    ))
                    active_call = None
                    # Will sell another covered call next iteration
                else:
                    # Called away - sell shares at strike
                    sale_proceeds = active_call["strike"] * SHARES_PER_CONTRACT
                    stock_pnl = (active_call["strike"] - cost_basis) * SHARES_PER_CONTRACT
                    total_pnl = active_call["premium_total"] + stock_pnl
                    cash += sale_proceeds

                    trades.append(BacktestTrade(
                        ticker=ticker, phase="covered_call",
                        entry_date=str(active_call["entry_date"]),
                        exit_date=str(date),
                        strike=active_call["strike"],
                        premium=active_call["premium_total"],
                        contracts=1, outcome="called_away",
                        pnl=total_pnl,
                        stock_price_entry=active_call["entry_price"],
                        stock_price_exit=price,
                    ))
                    shares_held = 0
                    cost_basis = 0.0
                    active_call = None

        i += 1

    # Calculate metrics
    total_premium = sum(t.premium for t in trades)
    total_pnl = sum(t.pnl for t in trades)
    final_equity = cash + shares_held * hist["Close"].iloc[-1] if len(hist) > 0 else cash
    num_trades = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = wins / num_trades if num_trades > 0 else 0

    # Max drawdown
    eq_values = [e["equity"] for e in equity_curve]
    peak = eq_values[0]
    max_dd = 0
    for v in eq_values:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    # Sharpe ratio (annualized)
    if len(eq_values) > 1:
        eq_returns = pd.Series(eq_values).pct_change().dropna()
        sharpe = eq_returns.mean() / eq_returns.std() * np.sqrt(252) if eq_returns.std() > 0 else 0
    else:
        sharpe = 0

    return BacktestResult(
        trades=trades,
        initial_capital=capital,
        final_capital=final_equity,
        total_premium=total_premium,
        total_pnl=total_pnl,
        num_trades=num_trades,
        win_rate=win_rate,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
        equity_curve=equity_curve,
    )


def backtest_portfolio(
    tickers: list[str],
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    capital: float = STARTING_CAPITAL,
    **kwargs,
) -> dict:
    """Backtest Wheel strategy across multiple tickers."""
    per_ticker_capital = capital / len(tickers)
    results = {}

    for ticker in tickers:
        try:
            result = backtest_wheel(
                ticker, start_date, end_date, per_ticker_capital, **kwargs
            )
            results[ticker] = result
            logger.info(f"{ticker}: {result.summary()}")
        except Exception as e:
            logger.error(f"Backtest failed for {ticker}: {e}")

    if not results:
        return {}

    # Aggregate
    total_pnl = sum(r.total_pnl for r in results.values())
    total_premium = sum(r.total_premium for r in results.values())
    final_capital = sum(r.final_capital for r in results.values())
    all_trades = [t for r in results.values() for t in r.trades]
    wins = sum(1 for t in all_trades if t.pnl > 0)

    return {
        "per_ticker": {k: v.summary() for k, v in results.items()},
        "aggregate": {
            "initial_capital": capital,
            "final_capital": round(final_capital, 2),
            "total_return_pct": round((final_capital / capital - 1) * 100, 2),
            "total_premium": round(total_premium, 2),
            "total_pnl": round(total_pnl, 2),
            "total_trades": len(all_trades),
            "win_rate": round(wins / len(all_trades) * 100, 2) if all_trades else 0,
        },
    }
