"""
IBKR Data Collector — fetches real-time and historical data for ML training.

Replaces yfinance with live IBKR data for:
  - Real bid/ask/IV/Greeks on options
  - Stock historical OHLCV
  - Account + P&L snapshots (for ML labels)
"""

import csv
import datetime as dt
import logging
import time
from pathlib import Path

import pandas as pd

from src.ibkr.connector import IBKRConnector

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

try:
    from src.data.fetcher import WHEEL_UNIVERSE
except ImportError:
    WHEEL_UNIVERSE = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]


class IBKRDataCollector:
    """
    Collects and saves market data from IBKR for ML training.

    Each day, call collect_daily_snapshot() to build up the dataset.
    """

    def __init__(self, ibkr: IBKRConnector):
        self.ibkr = ibkr

    # ── Stock snapshot ──────────────────────────────────────────────────────
    def get_stock_snapshot(self, ticker: str) -> dict:
        """Get current stock price + 20-day realized vol."""
        hist = self.ibkr.get_stock_history(ticker, duration="3 M", bar_size="1 day")
        if not hist:
            return {}

        closes = [b["close"] for b in hist if b["close"] > 0]
        if len(closes) < 21:
            return {}

        import numpy as np
        prices = pd.Series(closes)
        returns = prices.pct_change().dropna()
        rv20 = returns.tail(20).std() * (252 ** 0.5)
        rv60 = returns.tail(60).std() * (252 ** 0.5) if len(returns) >= 60 else rv20
        momentum_20 = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) >= 21 else 0

        # Simple IV rank proxy
        rolling_vol = returns.rolling(20).std() * (252 ** 0.5)
        iv_rank = float(
            (rv20 - rolling_vol.min()) / (rolling_vol.max() - rolling_vol.min()) * 100
        ) if rolling_vol.max() != rolling_vol.min() else 50.0

        # RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
        rsi = 100 - (100 / (1 + rs))

        current_price = prices.iloc[-1]
        sma20 = prices.tail(20).mean()
        sma50 = prices.tail(50).mean() if len(prices) >= 50 else sma20

        return {
            "date": dt.date.today().isoformat(),
            "ticker": ticker,
            "price": round(current_price, 2),
            "rv20": round(rv20, 4),
            "rv60": round(rv60, 4),
            "iv_rank": round(iv_rank, 2),
            "momentum_20d": round(momentum_20, 4),
            "rsi": round(rsi, 2),
            "above_sma20": int(current_price > sma20),
            "above_sma50": int(current_price > sma50),
            "sma20_above_sma50": int(sma20 > sma50),
        }

    # ── Option snapshot ─────────────────────────────────────────────────────
    def get_option_snapshot(
        self,
        ticker: str,
        expiration: str,
        strike: float,
        right: str = "P",
    ) -> dict:
        """
        Get real-time option data including Greeks.

        Args:
            expiration: YYYYMMDD format
            right: "P" put / "C" call
        """
        data = self.ibkr.get_option_price(ticker, expiration, strike, right)
        if not data:
            return {}

        exp_date = dt.datetime.strptime(expiration, "%Y%m%d").date()
        dte = (exp_date - dt.date.today()).days
        mid = ((data.get("bid", 0) or 0) + (data.get("ask", 0) or 0)) / 2

        return {
            "date": dt.date.today().isoformat(),
            "ticker": ticker,
            "expiration": expiration,
            "strike": strike,
            "right": right,
            "dte": dte,
            "bid": data.get("bid", 0),
            "ask": data.get("ask", 0),
            "mid": round(mid, 2),
            "iv": round(data.get("iv", 0) or 0, 4),
            "delta": round(data.get("delta", 0) or 0, 4),
            "gamma": round(data.get("gamma", 0) or 0, 4),
            "theta": round(data.get("theta", 0) or 0, 4),
            "vega": round(data.get("vega", 0) or 0, 4),
            "und_price": round(data.get("und_price", 0) or 0, 2),
            "premium_yield": round(mid / strike * 100, 3) if strike > 0 else 0,
        }

    # ── Active position tracker ─────────────────────────────────────────────
    def snapshot_active_positions(self, positions: list) -> list:
        """
        For each active Wheel position, record current market state.
        Used to label ML training data with outcomes.

        positions: list of Position objects from risk_manager
        """
        records = []
        for pos in positions:
            exp_fmt = pos.expiration.replace("-", "")
            right = "P" if pos.phase == "cash_secured_put" else "C"
            snap = self.get_option_snapshot(pos.ticker, exp_fmt, pos.strike, right)
            if snap:
                snap["phase"] = pos.phase
                snap["premium_collected"] = pos.premium_collected
                snap["entry_date"] = pos.entry_date
                records.append(snap)
        return records

    # ── Daily collection ────────────────────────────────────────────────────
    def collect_daily_snapshot(
        self,
        tickers: list = None,
        active_positions: list = None,
    ) -> dict:
        """
        Main daily collection routine.
        Call this once per trading day to build up the ML training dataset.
        """
        if tickers is None:
            tickers = WHEEL_UNIVERSE

        today = dt.date.today().isoformat()
        logger.info(f"=== Daily snapshot: {today} ===")

        # 1. Stock data for universe
        stock_records = []
        for ticker in tickers:
            logger.info(f"  Stock snapshot: {ticker}")
            snap = self.get_stock_snapshot(ticker)
            if snap:
                stock_records.append(snap)
            time.sleep(0.3)   # Be polite to the API

        self._append_csv(
            stock_records,
            DATA_DIR / "stock_snapshots.csv",
        )
        logger.info(f"  Saved {len(stock_records)} stock snapshots")

        # 2. Active position option data
        option_records = []
        if active_positions:
            option_records = self.snapshot_active_positions(active_positions)
            self._append_csv(
                option_records,
                DATA_DIR / "option_snapshots.csv",
            )
            logger.info(f"  Saved {len(option_records)} option snapshots")

        return {
            "date": today,
            "stock_snapshots": len(stock_records),
            "option_snapshots": len(option_records),
        }

    # ── Historical bulk collection ──────────────────────────────────────────
    def collect_stock_history(
        self,
        tickers: list = None,
        years: int = 5,
    ) -> pd.DataFrame:
        """
        One-time bulk collection of historical stock data.
        Saves to data/raw/stock_history.csv — use for ML feature engineering.
        """
        if tickers is None:
            tickers = WHEEL_UNIVERSE

        duration = f"{years} Y"
        all_bars = []

        for ticker in tickers:
            logger.info(f"Fetching {years}Y history: {ticker}")
            bars = self.ibkr.get_stock_history(ticker, duration=duration, bar_size="1 day")
            for bar in bars:
                bar["ticker"] = ticker
                all_bars.append(bar)
            time.sleep(1.0)   # Respect IBKR pacing limits

        df = pd.DataFrame(all_bars)
        if not df.empty:
            out = DATA_DIR / "stock_history.csv"
            df.to_csv(out, index=False)
            logger.info(f"Saved {len(df)} rows to {out}")

        return df

    # ── Utility ─────────────────────────────────────────────────────────────
    @staticmethod
    def _append_csv(records: list, filepath: Path):
        """Append records to a CSV file, creating headers if needed."""
        if not records:
            return
        filepath.parent.mkdir(parents=True, exist_ok=True)
        file_exists = filepath.exists()
        with open(filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(records)

    @staticmethod
    def load_snapshots() -> pd.DataFrame:
        """Load all collected stock snapshots for ML training."""
        path = DATA_DIR / "stock_snapshots.csv"
        if path.exists():
            return pd.read_csv(path, parse_dates=["date"])
        return pd.DataFrame()

    @staticmethod
    def load_option_snapshots() -> pd.DataFrame:
        """Load all collected option snapshots for ML training."""
        path = DATA_DIR / "option_snapshots.csv"
        if path.exists():
            return pd.read_csv(path, parse_dates=["date"])
        return pd.DataFrame()
