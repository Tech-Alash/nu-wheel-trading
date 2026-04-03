"""
Polygon.io Data Collector — real options data with actual IV and Greeks.

Free tier:  previous close, current snapshot, limited history
Starter ($29/mo): 2 years options history, full Greeks

Docs: https://polygon.io/docs/options
"""

import datetime as dt
import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient

load_dotenv()
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def get_client() -> RESTClient:
    key = os.getenv("POLYGON_API_KEY", "")
    if not key or key == "your_api_key_here":
        raise ValueError(
            "POLYGON_API_KEY not set!\n"
            "Edit .env file and add your key from https://polygon.io/dashboard/api-keys"
        )
    return RESTClient(key)


# ── 1. Options snapshot (current chain) ────────────────────────────────────

def get_options_snapshot(ticker: str, client: RESTClient = None) -> pd.DataFrame:
    """
    Get full options chain snapshot for a ticker — ALL expirations.
    Returns real bid/ask/IV/Greeks from Polygon.

    Available on FREE tier (previous day close).
    """
    if client is None:
        client = get_client()

    rows = []
    try:
        # Iterate through all option contracts for this ticker
        for opt in client.list_snapshot_options_chain(
            ticker,
            params={
                "limit": 250,
            }
        ):
            details = opt.details
            greeks  = opt.greeks
            day     = opt.day
            last_q  = opt.last_quote

            rows.append({
                "ticker":       ticker,
                "contract":     details.ticker if details else "",
                "expiration":   details.expiration_date if details else "",
                "strike":       details.strike_price if details else 0,
                "right":        details.contract_type if details else "",  # call/put
                "dte":          (dt.datetime.strptime(details.expiration_date, "%Y-%m-%d").date()
                                 - dt.date.today()).days if details and details.expiration_date else 0,
                # Real bid/ask
                "bid":          last_q.bid if last_q else 0,
                "ask":          last_q.ask if last_q else 0,
                "mid":          ((last_q.bid or 0) + (last_q.ask or 0)) / 2 if last_q else 0,
                # Real volume & OI
                "volume":       day.volume if day else 0,
                "open_interest": opt.open_interest or 0,
                # Real IV and Greeks
                "iv":           opt.implied_volatility or 0,
                "delta":        greeks.delta if greeks else 0,
                "gamma":        greeks.gamma if greeks else 0,
                "theta":        greeks.theta if greeks else 0,
                "vega":         greeks.vega if greeks else 0,
                # Price action
                "close":        day.close if day else 0,
                "vwap":         day.vwap if day else 0,
                "fetched_date": dt.date.today().isoformat(),
            })

    except Exception as e:
        logger.error(f"Error fetching options snapshot for {ticker}: {e}")

    df = pd.DataFrame(rows)
    logger.info(f"{ticker}: {len(df)} option contracts fetched")
    return df


# ── 2. Historical options OHLCV ─────────────────────────────────────────────

def get_option_history(
    contract_ticker: str,
    from_date: str,
    to_date: str,
    client: RESTClient = None,
) -> pd.DataFrame:
    """
    Get daily OHLCV for a specific options contract.
    e.g. contract_ticker = "O:AAPL241220P00150000"

    Requires Starter plan for full history.
    Free tier: last ~few months.
    """
    if client is None:
        client = get_client()

    rows = []
    try:
        for bar in client.list_aggs(
            contract_ticker,
            1, "day",
            from_date, to_date,
            adjusted=True,
            limit=500,
        ):
            rows.append({
                "contract": contract_ticker,
                "date":     dt.datetime.fromtimestamp(bar.timestamp / 1000).strftime("%Y-%m-%d"),
                "open":     bar.open,
                "high":     bar.high,
                "low":      bar.low,
                "close":    bar.close,
                "volume":   bar.volume,
                "vwap":     bar.vwap,
            })
    except Exception as e:
        logger.error(f"Error fetching history for {contract_ticker}: {e}")

    return pd.DataFrame(rows)


# ── 3. Stock aggregates ─────────────────────────────────────────────────────

def get_stock_history(
    ticker: str,
    from_date: str = "2022-01-01",
    to_date: str   = None,
    client: RESTClient = None,
) -> pd.DataFrame:
    """
    Get daily stock OHLCV from Polygon.
    Free tier: 2+ years of history available.
    """
    if client is None:
        client = get_client()
    if to_date is None:
        to_date = dt.date.today().isoformat()

    rows = []
    try:
        for bar in client.list_aggs(
            ticker, 1, "day",
            from_date, to_date,
            adjusted=True,
            limit=50000,
        ):
            rows.append({
                "ticker":  ticker,
                "date":    dt.datetime.fromtimestamp(bar.timestamp / 1000).strftime("%Y-%m-%d"),
                "open":    bar.open,
                "high":    bar.high,
                "low":     bar.low,
                "close":   bar.close,
                "volume":  bar.volume,
                "vwap":    bar.vwap,
            })
    except Exception as e:
        logger.error(f"Error fetching stock history for {ticker}: {e}")

    return pd.DataFrame(rows)


# ── 4. Daily snapshot collection ────────────────────────────────────────────

def collect_daily_options_snapshot(
    tickers: list,
    client: RESTClient = None,
    dte_min: int = 7,
    dte_max: int = 60,
) -> pd.DataFrame:
    """
    Collect end-of-day options snapshots for all tickers.
    Filter to our target DTE range (7-60 days).
    Saves to data/raw/options_snapshots_YYYY-MM-DD.csv
    """
    if client is None:
        client = get_client()

    today = dt.date.today().isoformat()
    all_rows = []

    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] Fetching options snapshot: {ticker}")
        df = get_options_snapshot(ticker, client)

        if not df.empty:
            # Filter to target DTE range and puts only
            df = df[
                (df["dte"] >= dte_min) &
                (df["dte"] <= dte_max) &
                (df["right"] == "put")
            ]
            all_rows.append(df)

        time.sleep(0.15)   # Respect rate limits

    if not all_rows:
        logger.warning("No data collected!")
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)

    # Save
    out = RAW_DIR / f"options_snapshots_{today}.csv"
    result.to_csv(out, index=False)
    logger.info(f"Saved {len(result)} option rows to {out}")

    return result


# ── 5. Build rich ML dataset with real IV ──────────────────────────────────

def build_ml_dataset_with_real_iv(
    stock_history_path: str = "data/raw/stock_history_ibkr.csv",
    snapshot_dir: str = "data/raw",
    output_path: str = "data/processed/ml_dataset_real_iv.csv",
) -> pd.DataFrame:
    """
    Combine IBKR stock history with Polygon real IV/Greeks snapshots
    to create a higher-quality ML training dataset.
    """
    from pathlib import Path
    import glob

    # Load all saved snapshots
    snapshots = []
    for f in glob.glob(f"{snapshot_dir}/options_snapshots_*.csv"):
        df = pd.read_csv(f)
        snapshots.append(df)

    if not snapshots:
        logger.warning("No option snapshots found. Run collect_daily_options_snapshot() first.")
        return pd.DataFrame()

    snap_df = pd.concat(snapshots, ignore_index=True)
    logger.info(f"Loaded {len(snap_df)} option snapshot rows from {len(snapshots)} days")

    # Merge with stock history
    stock_df = pd.read_csv(stock_history_path)
    stock_df["date"] = stock_df["date"].astype(str).str[:8].apply(
        lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}"
    )

    merged = snap_df.merge(
        stock_df[["ticker", "date", "close", "volume"]].rename(
            columns={"close": "stock_close", "volume": "stock_volume"}
        ),
        on=["ticker"],
        how="left",
    )

    out = Path(output_path)
    merged.to_csv(out, index=False)
    logger.info(f"Saved merged dataset: {len(merged)} rows to {out}")

    return merged


# ── 6. Quick test ────────────────────────────────────────────────────────────

def test_connection() -> bool:
    """Verify API key works and show account tier."""
    try:
        client = get_client()
        # Simple test: get previous close for SPY
        result = list(client.list_aggs("SPY", 1, "day", "2026-03-28", "2026-04-01", limit=5))
        if result:
            print(f"Polygon.io connected!")
            print(f"SPY last close: ${result[-1].close:.2f} on "
                  f"{dt.datetime.fromtimestamp(result[-1].timestamp/1000).strftime('%Y-%m-%d')}")

            # Test options
            print("\nTesting options data (MRNA)...")
            opts = list(client.list_snapshot_options_chain(
                "MRNA", params={"limit": 3}
            ))
            if opts:
                o = opts[0]
                print(f"  Contract: {o.details.ticker if o.details else 'N/A'}")
                print(f"  IV: {o.implied_volatility}")
                print(f"  Delta: {o.greeks.delta if o.greeks else 'N/A'}")
                print(f"  Bid/Ask: {o.last_quote.bid if o.last_quote else 0}"
                      f" / {o.last_quote.ask if o.last_quote else 0}")
            return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    test_connection()
