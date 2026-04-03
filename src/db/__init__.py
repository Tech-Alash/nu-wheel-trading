"""Database package — SQLite-backed portfolio persistence."""

from src.db.database import (
    init_db,
    load_portfolio,
    save_portfolio,
    record_trade,
    get_trade_history,
)

__all__ = [
    "init_db",
    "load_portfolio",
    "save_portfolio",
    "record_trade",
    "get_trade_history",
]
