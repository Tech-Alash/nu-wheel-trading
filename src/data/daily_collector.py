"""
Daily Options Data Collector
=============================
Собирает реальные данные опционов каждый торговый день.

Запускать в 17:30-22:00 по Алматы (рынок открыт).

Источники:
  - yfinance  → IV, volume, OI, strike, expiration (бесплатно, всегда)
  - IBKR API  → bid/ask, Greeks в реальном времени (только рынок открыт)
"""

import csv
import datetime as dt
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.data.fetcher import WHEEL_UNIVERSE
from src.models.strike_optimizer import (
    black_scholes_put_price,
    put_delta,
    probability_otm_put,
)

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _mid(bid, ask):
    if bid and ask and bid > 0 and ask > 0:
        return round((bid + ask) / 2, 3)
    return None


def _safe(val, default=0.0):
    try:
        v = float(val)
        return v if not np.isnan(v) else default
    except (TypeError, ValueError):
        return default


# ── Core: get options chain with real IV ─────────────────────────────────────

def get_options_with_iv(
    ticker: str,
    dte_min: int = 15,
    dte_max: int = 55,
    right: str = "put",
) -> pd.DataFrame:
    """
    Fetch options chain via yfinance — returns real IV for each contract.
    Works 24/7. bid/ask populated during market hours.
    """
    try:
        stk  = yf.Ticker(ticker)
        exps = stk.options
        if not exps:
            return pd.DataFrame()

        today = dt.date.today()
        rows  = []

        for exp_str in exps:
            exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if not (dte_min <= dte <= dte_max):
                continue

            chain = stk.option_chain(exp_str)
            df = chain.puts if right == "put" else chain.calls

            # Current stock price
            price = _safe(stk.fast_info.last_price) or _safe(stk.fast_info.previous_close)
            if price <= 0:
                continue

            for _, row in df.iterrows():
                strike = _safe(row.get("strike"))
                if strike <= 0:
                    continue

                iv     = _safe(row.get("impliedVolatility"), 0.30)
                bid    = _safe(row.get("bid"))
                ask    = _safe(row.get("ask"))
                mid    = _mid(bid, ask)
                volume = _safe(row.get("volume"))
                oi     = _safe(row.get("openInterest"))

                # If bid/ask unavailable → use BS with real IV
                if mid is None or mid <= 0:
                    T    = max(dte, 1) / 365
                    mid  = black_scholes_put_price(price, strike, T, 0.05, iv if iv > 0 else 0.30)
                    data_source = "bs_fallback"
                else:
                    data_source = "market"

                # Greeks via BS (yfinance doesn't provide them)
                T = max(dte, 1) / 365
                sig = iv if iv > 0 else 0.30
                delta_val = put_delta(price, strike, T, 0.05, sig)
                p_otm     = probability_otm_put(price, strike, T, 0.05, sig)
                otm_pct   = (price - strike) / price * 100 if price > strike else 0

                rows.append({
                    "date":          today.isoformat(),
                    "ticker":        ticker,
                    "expiration":    exp_str,
                    "dte":           dte,
                    "strike":        strike,
                    "stock_price":   round(price, 2),
                    "otm_pct":       round(otm_pct, 2),
                    "right":         right,
                    # Real market data
                    "iv":            round(iv, 4),
                    "bid":           bid,
                    "ask":           ask,
                    "mid":           round(mid, 3),
                    "volume":        int(volume),
                    "open_interest": int(oi),
                    # Derived
                    "delta":         round(delta_val, 4),
                    "prob_otm":      round(p_otm * 100, 2),
                    "premium_yield": round(mid / strike * 100, 3),
                    "annual_yield":  round(mid / strike * 100 / dte * 365, 2),
                    "data_source":   data_source,
                })

        return pd.DataFrame(rows)

    except Exception as e:
        logger.error(f"Error fetching options for {ticker}: {e}")
        return pd.DataFrame()


# ── Daily collection ──────────────────────────────────────────────────────────

def collect_daily_snapshot(
    tickers: list = None,
    dte_min: int = 15,
    dte_max: int = 55,
) -> pd.DataFrame:
    """
    Collect options snapshot for all tickers.
    Save to data/raw/options_YYYY-MM-DD.csv

    Best run at market close (~23:00 Alm) for bid/ask prices.
    """
    if tickers is None:
        tickers = WHEEL_UNIVERSE

    today  = dt.date.today().isoformat()
    output = RAW_DIR / f"options_{today}.csv"

    # Skip if already collected today
    if output.exists():
        logger.info(f"Already collected today: {output}")
        return pd.read_csv(output)

    all_rows = []
    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] {ticker}")
        df = get_options_with_iv(ticker, dte_min, dte_max)
        if not df.empty:
            all_rows.append(df)
        time.sleep(0.4)

    if not all_rows:
        logger.warning("No data collected!")
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)

    # Save
    result.to_csv(output, index=False)
    logger.info(f"Saved {len(result)} rows to {output}")
    return result


# ── Build ML dataset with real IV ────────────────────────────────────────────

def build_real_iv_dataset(
    snapshot_dir: str = "data/raw",
    stock_history: str = "data/raw/stock_history_ibkr.csv",
    output: str = "data/processed/ml_dataset_real_iv.csv",
) -> pd.DataFrame:
    """
    Merge accumulated daily snapshots into ML training dataset.
    Each row = one option contract on one day, with real IV.

    Requires at least a few days of collected snapshots.
    """
    import glob

    files = sorted(glob.glob(f"{snapshot_dir}/options_*.csv"))
    if not files:
        logger.error("No snapshots found. Run collect_daily_snapshot() first.")
        return pd.DataFrame()

    logger.info(f"Loading {len(files)} daily snapshots...")
    snaps = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    logger.info(f"Total rows: {len(snaps):,}")

    # Add stock technical features from IBKR history
    stock_df = pd.read_csv(stock_history)
    stock_df["date"] = stock_df["date"].astype(str).apply(
        lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}" if len(x) == 8 else x
    )

    # Compute rolling features per ticker
    tech_rows = []
    for ticker, g in stock_df.groupby("ticker"):
        g = g.sort_values("date").copy()
        c = g["close"]

        log_ret = np.log(c / c.shift(1))
        g["rv20"]       = log_ret.rolling(20).std() * np.sqrt(252)
        g["rv60"]       = log_ret.rolling(60).std() * np.sqrt(252)
        g["return_20d"] = c.pct_change(20)
        g["sma20"]      = c.rolling(20).mean()
        g["sma50"]      = c.rolling(50).mean()
        g["above_sma20"]= (c > g["sma20"]).astype(int)
        g["above_sma50"]= (c > g["sma50"]).astype(int)
        g["trend_score"]= g["above_sma20"] + g["above_sma50"] + (g["sma20"] > g["sma50"]).astype(int)
        tr = pd.concat([
            g["high"]-g["low"],
            (g["high"]-c.shift()).abs(),
            (g["low"]-c.shift()).abs(),
        ], axis=1).max(axis=1)
        g["atr_pct"] = tr.rolling(14).mean() / c * 100

        delta = c.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        g["rsi"] = 100 - (100 / (1 + rs))

        g["iv_rank"] = g["rv20"].rolling(252, min_periods=50).apply(
            lambda x: (x.iloc[-1]-x.min())/(x.max()-x.min())*100
            if x.max()!=x.min() else 50, raw=False
        )

        tech_rows.append(g[["ticker","date","rv20","rv60","return_20d",
                             "trend_score","rsi","atr_pct","iv_rank"]])

    tech_df = pd.concat(tech_rows, ignore_index=True)

    # Merge
    merged = snaps.merge(tech_df, on=["ticker","date"], how="left")

    # Add forward outcome: was it assigned? (only for puts)
    merged = merged[merged["right"] == "put"].copy()
    merged = _add_assignment_labels(merged, stock_df)

    out_path = Path(output)
    merged.to_csv(out_path, index=False)
    logger.info(f"Saved real-IV ML dataset: {len(merged):,} rows → {out_path}")
    return merged


def _add_assignment_labels(df: pd.DataFrame, stock_df: pd.DataFrame) -> pd.DataFrame:
    """Add was_assigned label based on stock price at expiration."""
    stock_pivot = stock_df.pivot_table(
        index="date", columns="ticker", values="close"
    )

    labels = []
    for _, row in df.iterrows():
        exp = row["expiration"]
        ticker = row["ticker"]
        strike = row["strike"]

        # Find closest trading day at/after expiration
        try:
            if exp in stock_pivot.index and ticker in stock_pivot.columns:
                price_at_exp = stock_pivot.loc[exp, ticker]
                was_assigned  = int(price_at_exp < strike) if not np.isnan(price_at_exp) else np.nan
            else:
                was_assigned = np.nan
        except Exception:
            was_assigned = np.nan

        labels.append(was_assigned)

    df["was_assigned"] = labels
    labeled = df.dropna(subset=["was_assigned"])
    logger.info(f"Labeled {len(labeled):,} / {len(df):,} rows "
                f"(assignment rate: {labeled['was_assigned'].mean()*100:.1f}%)")
    return df


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print("Collecting daily options snapshot...")
    result = collect_daily_snapshot(
        tickers=WHEEL_UNIVERSE,
        dte_min=15,
        dte_max=55,
    )

    if not result.empty:
        print(f"\nCollected {len(result):,} option rows")
        print(f"Market data: {(result.data_source=='market').sum():,} | BS fallback: {(result.data_source=='bs_fallback').sum():,}")
        print(f"\nTop puts by annual yield:")
        top = (
            result[result["right"]=="put"]
            .sort_values("annual_yield", ascending=False)
            .head(10)[["ticker","expiration","dte","strike","otm_pct","iv","mid","annual_yield","volume"]]
        )
        print(top.to_string(index=False))
