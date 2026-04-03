"""
Run this at 18:30 Almaty (market open) to collect fresh options data
and generate trading signals with real bid/ask prices.

Usage:
    python -X utf8 run_at_market_open.py
"""

import datetime as dt
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ── 0. Force-refresh today's snapshot ────────────────────────────────────────
today = dt.date.today().isoformat()
stale = Path(f"data/raw/options_{today}.csv")
if stale.exists():
    print(f"Removing stale pre-market snapshot: {stale}")
    stale.unlink()

# ── 1. Collect fresh options chain with real bid/ask ─────────────────────────
print("\n" + "="*60)
print("  STEP 1: Collecting live options data (real bid/ask)")
print("="*60)

from src.data.daily_collector import collect_daily_snapshot
from src.data.fetcher import WHEEL_UNIVERSE

result = collect_daily_snapshot(
    tickers=WHEEL_UNIVERSE,
    dte_min=15,
    dte_max=55,
)

if result.empty:
    print("ERROR: No data collected. Check internet connection.")
    exit(1)

by_src = result.data_source.value_counts()
print(f"\n  Collected {len(result):,} option rows")
print(f"  Real market data : {by_src.get('market', 0):,}")
print(f"  BS fallback      : {by_src.get('bs_fallback', 0):,}")

# ── 2. Run screener ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2: Running opportunity screener")
print("="*60)

from src.strategies.screener import screen_universe

signals = screen_universe(dte_min=20, dte_max=45, top_n=20)

if signals is not None and not signals.empty:
    print(f"\n  Top 15 opportunities:")
    cols = ["ticker", "expiration", "dte", "strike", "otm_pct",
            "iv_est", "mid_price", "annual_yield", "composite_score"]
    avail = [c for c in cols if c in signals.columns]
    print(signals[avail].head(15).to_string(index=False))
else:
    print("  No signals found.")

# ── 3. Generate trade signals ────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3: Generating trade signals (ML-assisted)")
print("="*60)

from src.strategies.signal_generator import generate_signals, print_signals
from src.db.database import load_portfolio

portfolio = load_portfolio()
print(f"\n  Portfolio: ${portfolio.cash:,.0f} cash | {len(portfolio.positions)} open positions")

trade_signals = generate_signals(portfolio)

if trade_signals:
    print_signals(trade_signals)
    print(f"\n  {len(trade_signals)} trade signals ready.")
    print("  To execute: python -X utf8 run_live.py")
else:
    print("  No trade signals at this time.")

print("\n" + "="*60)
print(f"  DONE — {dt.datetime.now().strftime('%H:%M:%S')}")
print("="*60)
