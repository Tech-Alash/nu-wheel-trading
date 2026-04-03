"""One-shot migration: JSON files -> SQLite.

Run ONCE to move existing data into the new DB.
Safe to run multiple times — it won't duplicate data.

Usage:
    python migrate_json_to_sqlite.py
"""

import json
import sys
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import STARTING_CAPITAL
from src.db.database import init_db, save_portfolio, record_trade, get_connection
from src.strategies.risk_manager import Portfolio, Position

OLD_PORTFOLIO_JSON = Path("data/portfolio_state.json")
OLD_TRADE_LOG_JSON = Path("data/trade_log.json")


def migrate_portfolio() -> int:
    """Migrate portfolio_state.json -> SQLite portfolio_state + positions tables."""
    if not OLD_PORTFOLIO_JSON.exists():
        print(f"  [SKIP] {OLD_PORTFOLIO_JSON} not found — nothing to migrate.")
        return 0

    with open(OLD_PORTFOLIO_JSON) as f:
        state = json.load(f)

    portfolio = Portfolio(
        cash=state.get("cash", STARTING_CAPITAL),
        total_premium_collected=state.get("total_premium_collected", 0.0),
        total_realized_pnl=state.get("total_realized_pnl", 0.0),
    )

    for p in state.get("positions", []):
        portfolio.positions.append(Position(**p))

    save_portfolio(portfolio)
    n = len(portfolio.positions)
    print(
        f"  [OK] Portfolio migrated: cash={portfolio.cash:.2f}, "
        f"{n} position(s)."
    )
    return n


def migrate_trade_log() -> int:
    """Migrate trade_log.json -> SQLite trade_log table (skip duplicates)."""
    if not OLD_TRADE_LOG_JSON.exists():
        print(f"  [SKIP] {OLD_TRADE_LOG_JSON} not found — nothing to migrate.")
        return 0

    with open(OLD_TRADE_LOG_JSON) as f:
        trades = json.load(f)

    if not trades:
        print("  [SKIP] Trade log JSON is empty.")
        return 0

    # Check if trade_log already has rows (avoid double-import)
    with get_connection() as conn:
        existing_count = conn.execute("SELECT COUNT(*) FROM trade_log").fetchone()[0]

    if existing_count > 0:
        print(
            f"  [SKIP] trade_log already has {existing_count} row(s). "
            "Not re-importing to avoid duplicates."
        )
        return 0

    count = 0
    for t in trades:
        record_trade(
            action=t.get("action", "UNKNOWN"),
            ticker=t.get("ticker", ""),
            strike=t.get("strike"),
            expiration=t.get("expiration"),
            premium=t.get("premium"),
            contracts=t.get("contracts"),
            total_premium=t.get("total_premium"),
            pnl=t.get("pnl"),
            notes=t.get("notes", ""),
        )
        count += 1

    print(f"  [OK] {count} trade log entry/entries migrated.")
    return count


def backup_json_files():
    """Rename old JSON files to *.bak so they are not accidentally used."""
    for path in (OLD_PORTFOLIO_JSON, OLD_TRADE_LOG_JSON):
        if path.exists():
            backup = path.with_suffix(".json.bak")
            path.rename(backup)
            print(f"  [BACKUP] {path} → {backup}")


def main():
    print("=" * 55)
    print("  JSON → SQLite Migration")
    print("=" * 55)

    print("\nStep 1: Initialising database …")
    init_db()

    print("\nStep 2: Migrating portfolio state …")
    migrate_portfolio()

    print("\nStep 3: Migrating trade log …")
    migrate_trade_log()

    print("\nStep 4: Backing up old JSON files …")
    backup_json_files()

    print("\n  Migration complete! DB is at data/portfolio.db")
    print("=" * 55)


if __name__ == "__main__":
    main()
