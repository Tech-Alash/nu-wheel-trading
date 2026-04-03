"""
Feature engineering for ML models.

Takes raw OHLCV history and builds:
  - Technical features (MA, RSI, vol, momentum)
  - Wheel strategy labels (was_assigned, optimal_close_day)
  - Simulated option premiums via Black-Scholes
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.models.strike_optimizer import black_scholes_put_price, probability_otm_put

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── Technical indicators ───────────────────────────────────────────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators per ticker."""
    results = []

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()
        close = g["close"]

        # Moving averages
        g["sma10"]  = close.rolling(10).mean()
        g["sma20"]  = close.rolling(20).mean()
        g["sma50"]  = close.rolling(50).mean()

        # Trend flags
        g["above_sma20"]       = (close > g["sma20"]).astype(int)
        g["above_sma50"]       = (close > g["sma50"]).astype(int)
        g["sma20_above_sma50"] = (g["sma20"] > g["sma50"]).astype(int)
        g["trend_score"]       = g["above_sma20"] + g["above_sma50"] + g["sma20_above_sma50"]

        # Returns
        g["return_1d"]  = close.pct_change(1)
        g["return_5d"]  = close.pct_change(5)
        g["return_20d"] = close.pct_change(20)

        # Realized volatility (annualised)
        log_ret = np.log(close / close.shift(1))
        g["rv10"]  = log_ret.rolling(10).std()  * np.sqrt(252)
        g["rv20"]  = log_ret.rolling(20).std()  * np.sqrt(252)
        g["rv60"]  = log_ret.rolling(60).std()  * np.sqrt(252)

        # IV proxy: RV × 1.15 (IV usually trades 15% above RV)
        g["iv_est"] = g["rv20"] * 1.15

        # IV Rank (0-100 percentile of current vol vs past year)
        g["iv_rank"] = (
            g["rv20"].rolling(252, min_periods=50)
            .apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100
                   if x.max() != x.min() else 50, raw=False)
        )

        # RSI-14
        delta = close.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        g["rsi"] = 100 - (100 / (1 + rs))

        # ATR-14 (average true range)
        tr = pd.concat([
            g["high"] - g["low"],
            (g["high"] - close.shift()).abs(),
            (g["low"]  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        g["atr14"] = tr.rolling(14).mean()
        g["atr_pct"] = g["atr14"] / close * 100   # As % of price

        # 52-week high/low distance
        g["dist_from_52w_high"] = (close.rolling(252, min_periods=50).max() - close) / close * 100
        g["dist_from_52w_low"]  = (close - close.rolling(252, min_periods=50).min()) / close * 100

        results.append(g)

    return pd.concat(results).reset_index(drop=True)


# ── Option simulation ──────────────────────────────────────────────────────

def simulate_wheel_outcomes(
    df: pd.DataFrame,
    otm_pct: float = 0.05,
    dte: int = 30,
    risk_free: float = 0.05,
) -> pd.DataFrame:
    """
    For each date in the dataset, simulate selling a cash-secured put:
      - strike = current_price * (1 - otm_pct)
      - premium via Black-Scholes
      - outcome: was the put assigned (stock < strike at expiry)?

    Also computes optimal early-close day (when 50% profit captured).
    """
    results = []
    T = dte / 365

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        closes = g["close"].values

        for i in range(len(g) - dte - 5):
            row = g.iloc[i]
            if pd.isna(row.get("iv_est")) or row["iv_est"] <= 0:
                continue

            S      = row["close"]
            sigma  = row["iv_est"]
            strike = round(S * (1 - otm_pct), 2)

            # Premium at entry
            premium = black_scholes_put_price(S, strike, T, risk_free, sigma)
            premium_yield = premium / strike * 100

            # Stock price at expiry
            S_exp = closes[i + dte]

            # Was it assigned?
            was_assigned = int(S_exp < strike)

            # P&L
            if was_assigned:
                pnl = premium - (strike - S_exp)   # Premium minus loss on stock
            else:
                pnl = premium                        # Keep full premium

            # Find day when 50% profit captured (for take-profit model)
            optimal_close_day = dte   # Default: hold to expiry
            for j in range(1, dte):
                if i + j >= len(closes):
                    break
                S_j     = closes[i + j]
                T_rem   = (dte - j) / 365
                curr_p  = black_scholes_put_price(S_j, strike, T_rem, risk_free, sigma)
                profit_pct = (premium - curr_p) / premium * 100
                if profit_pct >= 50:
                    optimal_close_day = j
                    break

            result = {
                "date":              row["date"],
                "ticker":            ticker,
                "stock_price":       S,
                "strike":            strike,
                "otm_pct":           otm_pct * 100,
                "dte":               dte,
                "premium":           round(premium, 3),
                "premium_yield":     round(premium_yield, 3),
                "iv_est":            round(sigma, 4),
                "iv_rank":           round(row.get("iv_rank", 50), 2),
                "rv20":              round(row.get("rv20", sigma), 4),
                "rsi":               round(row.get("rsi", 50), 2),
                "trend_score":       row.get("trend_score", 1),
                "above_sma20":       row.get("above_sma20", 1),
                "above_sma50":       row.get("above_sma50", 1),
                "return_20d":        round(row.get("return_20d", 0), 4),
                "atr_pct":           round(row.get("atr_pct", 2), 3),
                "dist_52w_high":     round(row.get("dist_from_52w_high", 10), 2),
                # Labels
                "was_assigned":      was_assigned,
                "pnl":               round(pnl, 3),
                "stock_at_expiry":   round(S_exp, 2),
                "optimal_close_day": optimal_close_day,
            }
            results.append(result)

    return pd.DataFrame(results)


# ── Main pipeline ──────────────────────────────────────────────────────────

def build_ml_dataset(
    history_path: str = "data/raw/stock_history_ibkr.csv",
    otm_pct: float = 0.05,
    dte: int = 30,
) -> pd.DataFrame:
    """Full pipeline: raw OHLCV → ML-ready dataset with labels."""

    print("Loading raw data...")
    df = pd.read_csv(history_path)
    df["date"] = df["date"].astype(str)
    print(f"  {len(df):,} rows, {df.ticker.nunique()} tickers")

    print("Computing technical features...")
    df = add_technical_features(df)
    print(f"  Features added: {[c for c in df.columns if c not in ['date','ticker','open','high','low','close','volume']]}")

    print(f"Simulating Wheel outcomes (OTM={otm_pct*100:.0f}%, DTE={dte})...")
    ml_df = simulate_wheel_outcomes(df, otm_pct=otm_pct, dte=dte)
    print(f"  {len(ml_df):,} training examples")
    print(f"  Assignment rate: {ml_df['was_assigned'].mean()*100:.1f}%")
    print(f"  Avg premium yield: {ml_df['premium_yield'].mean():.2f}%")
    print(f"  Win rate (OTM): {(1-ml_df['was_assigned'].mean())*100:.1f}%")

    # Save
    out = PROCESSED_DIR / f"ml_dataset_otm{int(otm_pct*100)}_dte{dte}.csv"
    ml_df.to_csv(out, index=False)
    print(f"  Saved to {out}")

    return ml_df


if __name__ == "__main__":
    # Build datasets for different OTM% and DTE combinations
    for otm in [0.03, 0.05, 0.08]:
        for dte in [21, 30, 45]:
            print(f"\n=== OTM={otm*100:.0f}% DTE={dte} ===")
            build_ml_dataset(otm_pct=otm, dte=dte)
    print("\nAll datasets built!")
