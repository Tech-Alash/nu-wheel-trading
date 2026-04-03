"""Data fetching module for stock and options data."""

import datetime as dt
import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import (
    MIN_MARKET_CAP,
    MIN_OPTION_VOLUME,
    MIN_STOCK_PRICE,
    MAX_STOCK_PRICE,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)


# Universe of liquid, optionable US stocks suitable for the Wheel strategy
# Focus on large-cap, high-volume names with weekly options
WHEEL_UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "TSLA",
    "INTC", "CRM", "ORCL", "ADBE", "NFLX", "PYPL", "SQ", "SHOP",
    "UBER", "COIN", "PLTR", "SNAP", "ROKU", "MARA",
    # Finance
    "JPM", "BAC", "GS", "MS", "C", "WFC", "SCHW", "COF",
    # Healthcare
    "JNJ", "PFE", "ABBV", "MRK", "UNH", "LLY", "BMY", "MRNA",
    # Consumer
    "WMT", "KO", "PEP", "MCD", "NKE", "SBUX", "DIS", "HD", "TGT",
    # Energy
    "XOM", "CVX", "OXY", "SLB", "DVN", "MPC",
    # Industrial
    "BA", "CAT", "DE", "GE", "F", "GM",
    # ETFs (high liquidity options)
    "SPY", "QQQ", "IWM", "EEM", "XLF", "XLE", "XLK", "GLD", "SLV",
    "TLT", "HYG", "ARKK",
]


def fetch_stock_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """Fetch historical OHLCV data for a stock."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return None
        df["Ticker"] = ticker
        return df
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return None


def fetch_options_chain(ticker: str) -> Optional[dict]:
    """Fetch full options chain with all expiration dates."""
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None

        chains = {}
        for exp in expirations:
            chain = stock.option_chain(exp)
            chains[exp] = {
                "calls": chain.calls,
                "puts": chain.puts,
            }
        return chains
    except Exception as e:
        logger.error(f"Error fetching options for {ticker}: {e}")
        return None


def fetch_stock_info(ticker: str) -> Optional[dict]:
    """Fetch fundamental info for a stock."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "ticker": ticker,
            "name": info.get("shortName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap", 0),
            "price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
            "avg_volume": info.get("averageVolume", 0),
            "beta": info.get("beta", 1.0),
            "dividend_yield": info.get("dividendYield", 0),
            "earnings_date": info.get("earningsTimestamp", None),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
        }
    except Exception as e:
        logger.error(f"Error fetching info for {ticker}: {e}")
        return None


def compute_iv_rank(ticker: str, period: str = "1y") -> Optional[float]:
    """Compute IV Rank (percentile of current IV vs past year)."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty or len(hist) < 20:
            return None

        # Use realized volatility as proxy when IV history isn't available
        returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
        if rolling_vol.dropna().empty:
            return None

        current_vol = rolling_vol.iloc[-1]
        vol_min = rolling_vol.min()
        vol_max = rolling_vol.max()

        if vol_max == vol_min:
            return 50.0

        iv_rank = ((current_vol - vol_min) / (vol_max - vol_min)) * 100
        return round(iv_rank, 2)
    except Exception as e:
        logger.error(f"Error computing IV rank for {ticker}: {e}")
        return None


def fetch_universe_data(tickers: list[str] = None) -> pd.DataFrame:
    """Fetch summary data for all tickers in the universe."""
    if tickers is None:
        tickers = WHEEL_UNIVERSE

    results = []
    for ticker in tickers:
        logger.info(f"Fetching data for {ticker}...")
        info = fetch_stock_info(ticker)
        if info is None:
            continue

        iv_rank = compute_iv_rank(ticker)
        info["iv_rank"] = iv_rank
        results.append(info)

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Filter based on criteria
    df = df[
        (df["price"] >= MIN_STOCK_PRICE)
        & (df["price"] <= MAX_STOCK_PRICE)
        & (df["market_cap"] >= MIN_MARKET_CAP)
    ]

    return df.sort_values("iv_rank", ascending=False).reset_index(drop=True)


def get_near_term_puts(ticker: str, dte_min: int = 20, dte_max: int = 45) -> Optional[pd.DataFrame]:
    """Get put options within the target DTE range."""
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        today = dt.date.today()

        target_puts = []
        for exp_str in expirations:
            exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days

            if dte_min <= dte <= dte_max:
                chain = stock.option_chain(exp_str)
                puts = chain.puts.copy()
                puts["expiration"] = exp_str
                puts["dte"] = dte
                puts["ticker"] = ticker
                target_puts.append(puts)

        if not target_puts:
            return None

        return pd.concat(target_puts, ignore_index=True)
    except Exception as e:
        logger.error(f"Error fetching puts for {ticker}: {e}")
        return None


def get_near_term_calls(ticker: str, dte_min: int = 20, dte_max: int = 45) -> Optional[pd.DataFrame]:
    """Get call options within the target DTE range for covered calls."""
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        today = dt.date.today()

        target_calls = []
        for exp_str in expirations:
            exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days

            if dte_min <= dte <= dte_max:
                chain = stock.option_chain(exp_str)
                calls = chain.calls.copy()
                calls["expiration"] = exp_str
                calls["dte"] = dte
                calls["ticker"] = ticker
                target_calls.append(calls)

        if not target_calls:
            return None

        return pd.concat(target_calls, ignore_index=True)
    except Exception as e:
        logger.error(f"Error fetching calls for {ticker}: {e}")
        return None
