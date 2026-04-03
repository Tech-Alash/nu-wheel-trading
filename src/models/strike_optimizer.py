"""ML-based strike price and DTE optimizer for the Wheel strategy.

Uses features like IV, technical indicators, and historical outcomes
to predict optimal strike selection that maximizes premium while
minimizing assignment risk (or maximizing favorable assignment).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar

from src.config import (
    DEFAULT_DTE_MIN,
    DEFAULT_DTE_MAX,
    DEFAULT_DELTA_PUT,
    DEFAULT_DELTA_CALL,
    SHARES_PER_CONTRACT,
)

logger = logging.getLogger(__name__)


def black_scholes_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes put option price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


def black_scholes_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes call option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate put option delta."""
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1


def call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate call option delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def probability_otm_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Probability that put expires OTM (we keep premium)."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0

    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d2)


def expected_return_put(
    S: float, K: float, T: float, r: float, sigma: float, premium: float
) -> float:
    """Expected return from selling a cash-secured put.

    Considers:
    - Probability of expiring OTM (keep full premium)
    - Probability of assignment (own stock at strike - premium)
    - Expected loss if assigned
    """
    p_otm = probability_otm_put(S, K, T, r, sigma)

    # If OTM: profit = premium
    profit_otm = premium

    # If ITM: expected loss = E[K - S_T | S_T < K] - premium
    # Use conditional expectation under lognormal
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Expected value of (K - S_T) given S_T < K
    expected_stock_if_itm = S * np.exp(r * T) * norm.cdf(-d1) / norm.cdf(-d2) if norm.cdf(-d2) > 0.001 else K
    expected_loss_if_assigned = K - expected_stock_if_itm

    profit_itm = premium - expected_loss_if_assigned

    expected_return = p_otm * profit_otm + (1 - p_otm) * profit_itm
    # Annualize
    annual_return = (expected_return / K) * (365 / (T * 365))

    return annual_return


def optimize_put_strike(
    stock_price: float,
    sigma: float,
    dte: int,
    risk_free_rate: float = 0.05,
    min_otm_pct: float = 0.03,
    max_otm_pct: float = 0.15,
) -> dict:
    """Find optimal strike for cash-secured put.

    Balances premium income vs assignment risk.
    """
    T = dte / 365

    best_result = None
    best_score = -np.inf

    # Search over strike prices
    for otm_pct in np.arange(min_otm_pct, max_otm_pct + 0.005, 0.005):
        K = round(stock_price * (1 - otm_pct), 2)

        premium = black_scholes_put_price(stock_price, K, T, risk_free_rate, sigma)
        delta = put_delta(stock_price, K, T, risk_free_rate, sigma)
        p_otm = probability_otm_put(stock_price, K, T, risk_free_rate, sigma)
        exp_return = expected_return_put(stock_price, K, T, risk_free_rate, sigma, premium)

        # Premium yield (annualized)
        premium_yield = (premium / K) * (365 / dte) * 100

        # Composite score: maximize expected return with penalty for high assignment risk
        score = exp_return * 100 + p_otm * 10

        if score > best_score:
            best_score = score
            best_result = {
                "strike": K,
                "otm_pct": round(otm_pct * 100, 2),
                "premium": round(premium, 2),
                "premium_yield_annual": round(premium_yield, 2),
                "delta": round(delta, 4),
                "prob_otm": round(p_otm * 100, 2),
                "expected_return_annual": round(exp_return * 100, 2),
                "dte": dte,
                "score": round(best_score, 4),
            }

    return best_result


def optimize_call_strike(
    stock_price: float,
    cost_basis: float,
    sigma: float,
    dte: int,
    risk_free_rate: float = 0.05,
    min_otm_pct: float = 0.02,
    max_otm_pct: float = 0.10,
) -> dict:
    """Find optimal strike for covered call after assignment.

    Tries to sell at or above cost basis while maximizing premium.
    """
    T = dte / 365

    best_result = None
    best_score = -np.inf

    for otm_pct in np.arange(min_otm_pct, max_otm_pct + 0.005, 0.005):
        K = round(stock_price * (1 + otm_pct), 2)

        premium = black_scholes_call_price(stock_price, K, T, risk_free_rate, sigma)
        # Adjust: we want call premium (intrinsic + extrinsic for OTM calls)
        premium = max(premium, 0.01)

        delta_val = call_delta(stock_price, K, T, risk_free_rate, sigma)

        # Probability of being called away
        p_itm = norm.cdf(
            (np.log(stock_price / K) + (risk_free_rate + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        )

        premium_yield = (premium / stock_price) * (365 / dte) * 100

        # Prefer strikes above cost basis
        above_cost_bonus = 20 if K >= cost_basis else -10

        # Score: balance premium with not getting called away too easily
        score = premium_yield * 0.5 + (1 - p_itm) * 100 * 0.3 + above_cost_bonus

        if score > best_score:
            best_score = score
            best_result = {
                "strike": K,
                "otm_pct": round(otm_pct * 100, 2),
                "premium": round(premium, 2),
                "premium_yield_annual": round(premium_yield, 2),
                "delta": round(delta_val, 4),
                "prob_called_away": round(p_itm * 100, 2),
                "above_cost_basis": K >= cost_basis,
                "dte": dte,
                "score": round(best_score, 4),
            }

    return best_result


def find_optimal_dte(
    stock_price: float,
    sigma: float,
    risk_free_rate: float = 0.05,
    dte_range: tuple = (DEFAULT_DTE_MIN, DEFAULT_DTE_MAX),
) -> dict:
    """Find the optimal DTE for selling puts based on theta decay curve."""
    results = []

    for dte in range(dte_range[0], dte_range[1] + 1):
        opt = optimize_put_strike(stock_price, sigma, dte, risk_free_rate)
        if opt:
            results.append(opt)

    if not results:
        return {}

    # Best expected return per day of capital locked up
    df = pd.DataFrame(results)
    df["return_per_day"] = df["expected_return_annual"] / 365
    df["efficiency"] = df["premium"] / df["dte"]  # Premium earned per day

    best_idx = df["efficiency"].idxmax()
    return df.iloc[best_idx].to_dict()
