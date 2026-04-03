"""Stock screener for identifying best Wheel strategy candidates.

Ranks stocks by a composite score combining:
- Premium yield (higher IV = more premium)
- Downside protection (technical support, fundamentals)
- Liquidity (option volume, bid-ask spread)
- Trend strength (avoid selling puts into falling knives)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.models.strike_optimizer import black_scholes_put_price
from src.config import (
    SHARES_PER_CONTRACT,
    STARTING_CAPITAL,
    MAX_POSITION_PCT,
    DEFAULT_DTE_MIN,
    DEFAULT_DTE_MAX,
    DEFAULT_DELTA_PUT,
    EARNINGS_BUFFER_DAYS,
)
from src.data.fetcher import (
    fetch_stock_data,
    fetch_stock_info,
    compute_iv_rank,
    get_near_term_puts,
    WHEEL_UNIVERSE,
)

logger = logging.getLogger(__name__)


def compute_technical_score(df: pd.DataFrame) -> dict:
    """Compute technical indicators for trend assessment."""
    if df is None or len(df) < 50:
        return {"trend_score": 0, "support_distance": 0, "rsi": 50}

    close = df["Close"]

    # Moving averages
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    current = close.iloc[-1]

    # Trend score: above MAs = bullish
    trend_score = 0
    if current > sma20:
        trend_score += 1
    if current > sma50:
        trend_score += 1
    if sma20 > sma50:
        trend_score += 1

    # Distance from 52-week low (higher = more room to fall but safer)
    low_52w = close.rolling(252).min().iloc[-1] if len(close) >= 252 else close.min()
    support_distance = (current - low_52w) / current * 100

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
    rsi = 100 - (100 / (1 + rs))

    # Volatility (20-day realized)
    returns = np.log(close / close.shift(1)).dropna()
    realized_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100

    return {
        "trend_score": trend_score,       # 0-3 (higher = stronger uptrend)
        "support_distance": round(support_distance, 2),
        "rsi": round(rsi, 2),
        "realized_vol": round(realized_vol, 2),
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2),
        "current_price": round(current, 2),
    }


def compute_premium_score(ticker: str, stock_price: float) -> dict:
    """Evaluate premium quality for puts at target delta."""
    puts = get_near_term_puts(ticker, DEFAULT_DTE_MIN, DEFAULT_DTE_MAX)
    if puts is None or puts.empty:
        return {"premium_yield": 0, "best_strike": 0, "best_expiry": "", "best_premium": 0}

    # Find puts near our target delta (OTM puts with ~30 delta)
    # Approximate: look for strikes at ~5-10% below current price
    target_strike_low = stock_price * 0.85
    target_strike_high = stock_price * 0.97

    candidates = puts[
        (puts["strike"] >= target_strike_low)
        & (puts["strike"] <= target_strike_high)
        & (puts["volume"].fillna(0) > 0)
    ].copy()

    if candidates.empty:
        # Broaden the search
        candidates = puts[
            (puts["strike"] >= stock_price * 0.80)
            & (puts["strike"] <= stock_price * 0.99)
        ].copy()

    if candidates.empty:
        return {"premium_yield": 0, "best_strike": 0, "best_expiry": "", "best_premium": 0}

    # Use mid price as fair value; fall back to Black-Scholes when bid/ask are zero
    candidates["mid_price"] = (candidates["bid"].fillna(0) + candidates["ask"].fillna(0)) / 2

    # Estimate realized vol for BS fallback
    from src.data.fetcher import fetch_stock_data
    hist = fetch_stock_data(ticker, period="3mo")
    sigma = 0.30  # Default
    if hist is not None and len(hist) > 20:
        rets = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        sigma = rets.rolling(20).std().iloc[-1] * np.sqrt(252) * 1.15  # IV premium

    def bs_fallback(row):
        if row["mid_price"] > 0.01:
            return row["mid_price"]
        T = max(row["dte"], 1) / 365
        return black_scholes_put_price(stock_price, row["strike"], T, 0.05, sigma)

    candidates["mid_price"] = candidates.apply(bs_fallback, axis=1)

    candidates["premium_yield"] = (
        candidates["mid_price"] / candidates["strike"] * 100
    )
    # Annualize the yield
    candidates["annual_yield"] = (
        candidates["premium_yield"] / candidates["dte"] * 365
    )

    # Pick the best risk/reward: highest annualized yield with decent OTM distance
    candidates["otm_pct"] = (stock_price - candidates["strike"]) / stock_price * 100
    candidates["composite"] = candidates["annual_yield"] * 0.6 + candidates["otm_pct"] * 0.4

    best = candidates.sort_values("composite", ascending=False).iloc[0]

    return {
        "premium_yield": round(best["annual_yield"], 2),
        "best_strike": best["strike"],
        "best_expiry": best["expiration"],
        "best_premium": round(best["mid_price"], 2),
        "best_dte": int(best["dte"]),
        "otm_pct": round(best["otm_pct"], 2),
    }


def screen_universe(
    tickers: list[str] = None,
    top_n: int = 10,
    portfolio_value: float = STARTING_CAPITAL,
) -> pd.DataFrame:
    """Screen the full universe and rank stocks for Wheel strategy.

    Returns a DataFrame ranked by composite score with trade recommendations.
    """
    if tickers is None:
        tickers = WHEEL_UNIVERSE

    results = []

    for ticker in tickers:
        logger.info(f"Screening {ticker}...")

        # Fetch price data
        hist = fetch_stock_data(ticker, period="1y")
        if hist is None or hist.empty:
            continue

        current_price = hist["Close"].iloc[-1]

        # Check if position size fits within limits
        position_cost = current_price * SHARES_PER_CONTRACT
        max_position = portfolio_value * MAX_POSITION_PCT
        if position_cost > max_position:
            logger.info(f"Skipping {ticker}: position size ${position_cost:.0f} > max ${max_position:.0f}")
            continue

        # Technical analysis
        technicals = compute_technical_score(hist)

        # Premium analysis
        premium_info = compute_premium_score(ticker, current_price)

        # IV Rank
        iv_rank = compute_iv_rank(ticker)

        # Stock info
        info = fetch_stock_info(ticker)

        # Composite score
        # Weight: premium 35%, trend 25%, IV rank 20%, liquidity 10%, fundamentals 10%
        score = 0.0

        # Premium component (0-100)
        premium_score = min(premium_info.get("premium_yield", 0), 100)
        score += premium_score * 0.35

        # Trend component (0-100): uptrend good for selling puts
        trend_normalized = technicals["trend_score"] / 3 * 100
        # Penalize very high RSI (overbought) and very low RSI (falling knife)
        rsi = technicals["rsi"]
        if rsi < 30:
            trend_normalized *= 0.5   # Falling knife penalty
        elif rsi > 70:
            trend_normalized *= 0.8   # Overbought slight penalty
        score += trend_normalized * 0.25

        # IV Rank component (0-100): higher = better premiums
        if iv_rank is not None:
            score += iv_rank * 0.20

        # Liquidity (simplified)
        if info and info.get("avg_volume", 0) > 5_000_000:
            score += 100 * 0.10
        elif info and info.get("avg_volume", 0) > 1_000_000:
            score += 60 * 0.10

        # Fundamentals (beta preference: 0.8-1.5 is sweet spot)
        beta = info.get("beta", 1.0) if info else 1.0
        if beta and 0.8 <= beta <= 1.5:
            score += 100 * 0.10
        elif beta and 0.5 <= beta <= 2.0:
            score += 50 * 0.10

        result = {
            "ticker": ticker,
            "price": round(current_price, 2),
            "position_cost": round(position_cost, 0),
            "sector": info.get("sector", "") if info else "",
            "iv_rank": iv_rank,
            "trend_score": technicals["trend_score"],
            "rsi": technicals["rsi"],
            "realized_vol": technicals.get("realized_vol", 0),
            "premium_yield_annual": premium_info.get("premium_yield", 0),
            "best_strike": premium_info.get("best_strike", 0),
            "best_expiry": premium_info.get("best_expiry", ""),
            "best_premium": premium_info.get("best_premium", 0),
            "best_dte": premium_info.get("best_dte", 0),
            "otm_pct": premium_info.get("otm_pct", 0),
            "composite_score": round(score, 2),
        }
        results.append(result)

    df = pd.DataFrame(results)
    if df.empty:
        return df

    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df.head(top_n)
