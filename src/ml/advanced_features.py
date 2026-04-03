"""
Advanced Feature Engineering for Wheel Strategy ML Models
==========================================================
Builds 40+ features from stock OHLCV data:
  - Technical indicators (MA, RSI, MACD, Bollinger, Stochastic)
  - Volatility regime features (RV ratios, vol-of-vol, skew)
  - Momentum & mean-reversion signals
  - Volume/liquidity features
  - Cross-asset / market regime (VIX proxy, SPY correlation)
  - Interaction features (IV_rank * trend, RSI * vol, etc.)
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ── Core Technical Indicators ──────────────────────────────────────────────

def add_moving_averages(g: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """EMA and SMA at multiple windows."""
    for w in [5, 10, 20, 50, 100, 200]:
        g[f"sma{w}"] = close.rolling(w).mean()
        g[f"ema{w}"] = close.ewm(span=w, adjust=False).mean()

    # Price relative to MAs (normalized)
    g["price_vs_sma20"] = (close - g["sma20"]) / g["sma20"] * 100
    g["price_vs_sma50"] = (close - g["sma50"]) / g["sma50"] * 100
    g["price_vs_sma200"] = (close - g["sma200"]) / g["sma200"] * 100

    # Trend score (0-5)
    g["above_sma20"] = (close > g["sma20"]).astype(int)
    g["above_sma50"] = (close > g["sma50"]).astype(int)
    g["above_sma200"] = (close > g["sma200"]).astype(int)
    g["sma20_above_sma50"] = (g["sma20"] > g["sma50"]).astype(int)
    g["sma50_above_sma200"] = (g["sma50"] > g["sma200"]).astype(int)
    g["trend_score"] = (
        g["above_sma20"] + g["above_sma50"] + g["above_sma200"]
        + g["sma20_above_sma50"] + g["sma50_above_sma200"]
    )
    return g


def add_momentum(g: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """RSI, MACD, Stochastic, Rate of Change."""
    # Returns at multiple horizons
    for d in [1, 3, 5, 10, 20, 60]:
        g[f"return_{d}d"] = close.pct_change(d)

    # RSI-14
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    g["rsi"] = 100 - (100 / (1 + rs))

    # RSI divergence (price up but RSI down = bearish divergence)
    g["rsi_5d_change"] = g["rsi"].diff(5)
    g["price_rsi_divergence"] = g["return_5d"] * 100 - g["rsi_5d_change"]

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    g["macd"] = ema12 - ema26
    g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
    g["macd_hist"] = g["macd"] - g["macd_signal"]
    g["macd_bullish"] = (g["macd"] > g["macd_signal"]).astype(int)

    # Stochastic %K, %D
    low14 = g["low"].rolling(14).min()
    high14 = g["high"].rolling(14).max()
    g["stoch_k"] = (close - low14) / (high14 - low14 + 1e-8) * 100
    g["stoch_d"] = g["stoch_k"].rolling(3).mean()

    # Rate of change
    g["roc_10"] = (close / close.shift(10) - 1) * 100
    g["roc_20"] = (close / close.shift(20) - 1) * 100

    # Consecutive up/down days
    g["up_day"] = (close > close.shift(1)).astype(int)
    g["consec_up"] = g["up_day"].groupby(
        (g["up_day"] != g["up_day"].shift()).cumsum()
    ).cumsum() * g["up_day"]
    g["consec_down"] = (1 - g["up_day"]).groupby(
        ((1 - g["up_day"]) != (1 - g["up_day"]).shift()).cumsum()
    ).cumsum() * (1 - g["up_day"])

    return g


def add_volatility(g: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Realized vol, ATR, Bollinger, vol regime features."""
    log_ret = np.log(close / close.shift(1))

    # Realized vol at multiple windows
    for w in [5, 10, 20, 60]:
        g[f"rv{w}"] = log_ret.rolling(w).std() * np.sqrt(252)

    # IV estimate (RV20 * 1.15 markup)
    g["iv_est"] = g["rv20"] * 1.15

    # Vol of vol (volatility clustering)
    g["vol_of_vol"] = g["rv20"].rolling(20).std()

    # Vol ratio (short-term vs long-term — spikes indicate fear)
    g["rv_ratio_5_20"] = g["rv5"] / g["rv20"].replace(0, np.nan)
    g["rv_ratio_10_60"] = g["rv10"] / g["rv60"].replace(0, np.nan)

    # IV Rank (percentile over 1 year)
    g["iv_rank"] = g["rv20"].rolling(252, min_periods=50).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100
        if x.max() != x.min() else 50, raw=False
    )

    # IV percentile (% of days below current)
    g["iv_percentile"] = g["rv20"].rolling(252, min_periods=50).apply(
        lambda x: (x < x.iloc[-1]).mean() * 100, raw=False
    )

    # ATR
    tr = pd.concat([
        g["high"] - g["low"],
        (g["high"] - close.shift()).abs(),
        (g["low"] - close.shift()).abs(),
    ], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14).mean()
    g["atr_pct"] = g["atr14"] / close * 100

    # Bollinger Bands
    bb_sma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    g["bb_upper"] = bb_sma + 2 * bb_std
    g["bb_lower"] = bb_sma - 2 * bb_std
    g["bb_width"] = (g["bb_upper"] - g["bb_lower"]) / bb_sma * 100
    g["bb_position"] = (close - g["bb_lower"]) / (g["bb_upper"] - g["bb_lower"] + 1e-8)

    # 52-week extremes
    g["dist_52w_high"] = (close.rolling(252, min_periods=50).max() - close) / close * 100
    g["dist_52w_low"] = (close - close.rolling(252, min_periods=50).min()) / close * 100

    # Realized skewness (3rd moment of returns)
    g["return_skew_20"] = log_ret.rolling(20).skew()

    # Realized kurtosis (tail risk)
    g["return_kurt_20"] = log_ret.rolling(20).kurt()

    return g


def add_volume_features(g: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Volume-based features."""
    vol = g["volume"]

    # Relative volume
    g["volume_sma20"] = vol.rolling(20).mean()
    g["relative_volume"] = vol / g["volume_sma20"].replace(0, np.nan)

    # Volume trend
    g["volume_trend"] = (
        vol.rolling(5).mean() / vol.rolling(20).mean().replace(0, np.nan)
    )

    # On-Balance Volume (OBV) momentum
    obv = (np.sign(close.diff()) * vol).cumsum()
    g["obv_5d_change"] = obv.pct_change(5) * 100

    # Money Flow Index (MFI) - RSI with volume
    typical_price = (g["high"] + g["low"] + close) / 3
    money_flow = typical_price * vol
    tp_diff = typical_price.diff()
    pos_flow = money_flow.where(tp_diff > 0, 0).rolling(14).sum()
    neg_flow = money_flow.where(tp_diff <= 0, 0).rolling(14).sum()
    g["mfi"] = 100 - (100 / (1 + pos_flow / neg_flow.replace(0, np.nan)))

    return g


# ── Interaction & Derived Features ─────────────────────────────────────────

def add_interaction_features(g: pd.DataFrame) -> pd.DataFrame:
    """Cross-feature interactions that capture regime dynamics."""
    # IV * Trend (high IV + downtrend = dangerous for puts)
    g["iv_x_trend"] = g.get("iv_rank", 50) * g.get("trend_score", 2.5) / 5

    # RSI extremes * Volatility (oversold + high vol = bounce likely)
    g["rsi_x_vol"] = (50 - g.get("rsi", 50).abs()) * g.get("rv20", 0.3)

    # Momentum acceleration (2nd derivative of price)
    g["momentum_accel"] = g.get("return_5d", 0) - g.get("return_5d", 0).shift(5)

    # Vol regime change speed
    g["vol_momentum"] = g.get("rv5", 0.3) - g.get("rv20", 0.3)

    # Support proximity (how close to Bollinger lower band)
    g["support_distance"] = g.get("bb_position", 0.5)

    # Fear indicator: high vol + downtrend + high volume
    g["fear_score"] = (
        (g.get("rv_ratio_5_20", 1) - 1).clip(0, 3) * 33
        + (1 - g.get("trend_score", 2.5) / 5) * 33
        + (g.get("relative_volume", 1) - 1).clip(0, 3) * 33
    )

    return g


# ── Full Pipeline ──────────────────────────────────────────────────────────

def build_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Input: raw OHLCV with columns [date, ticker, open, high, low, close, volume]
    Output: enriched DataFrame with 50+ features per row.
    """
    results = []

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()
        close = g["close"]

        g = add_moving_averages(g, close)
        g = add_momentum(g, close)
        g = add_volatility(g, close)
        g = add_volume_features(g, close)
        g = add_interaction_features(g)

        results.append(g)

    result = pd.concat(results).reset_index(drop=True)
    return result


# ── Feature sets for different models ──────────────────────────────────────

# Minimal (8 features, fast, good baseline)
FEATURES_MINIMAL = [
    "premium_yield", "iv_est", "iv_rank", "rv20",
    "rsi", "trend_score", "return_20d", "atr_pct",
]

# Standard (20 features, balanced)
FEATURES_STANDARD = FEATURES_MINIMAL + [
    "rv5", "rv60", "rv_ratio_5_20",
    "return_5d", "return_60d",
    "bb_position", "bb_width",
    "macd_bullish", "stoch_k",
    "relative_volume", "dist_52w_high",
    "vol_of_vol",
]

# Full (35+ features, max info, risk of overfitting)
FEATURES_FULL = FEATURES_STANDARD + [
    "iv_percentile", "rv_ratio_10_60",
    "return_skew_20", "return_kurt_20",
    "roc_10", "roc_20",
    "mfi", "obv_5d_change",
    "volume_trend", "consec_up", "consec_down",
    "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
    "iv_x_trend", "fear_score", "vol_momentum",
    "momentum_accel", "support_distance",
    "price_rsi_divergence",
]
