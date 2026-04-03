"""SQLite-backed portfolio persistence layer.

Replaces the fragile JSON-file approach with proper ACID transactions.
Uses only stdlib sqlite3 — no extra dependencies needed.

DB lives at: data/portfolio.db
Tables:
  - portfolio_state : one row with current cash/P&L totals
  - positions       : active/closed option positions
  - trade_log       : append-only audit trail of every trade event
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from src.config import STARTING_CAPITAL
from src.strategies.risk_manager import Portfolio, Position

logger = logging.getLogger(__name__)

DB_PATH = Path("data/portfolio.db")


# ---------------------------------------------------------------------------
# Connection / context manager
# ---------------------------------------------------------------------------

@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection with WAL mode and foreign keys enabled.

    WAL (Write-Ahead Logging) mode is critical: it allows readers to continue
    while a writer is active (no full-file lock on every write).
    The connection is committed on clean exit and rolled back on exception.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("BEGIN")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio_state (
    id                      INTEGER PRIMARY KEY CHECK (id = 1),  -- singleton row
    cash                    REAL    NOT NULL DEFAULT 0.0,
    total_premium_collected REAL    NOT NULL DEFAULT 0.0,
    total_realized_pnl      REAL    NOT NULL DEFAULT 0.0,
    updated_at              TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS positions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker            TEXT    NOT NULL,
    phase             TEXT    NOT NULL,   -- cash_secured_put | assigned | covered_call
    entry_date        TEXT    NOT NULL,
    strike            REAL    NOT NULL,
    expiration        TEXT    NOT NULL,
    premium_collected REAL    NOT NULL DEFAULT 0.0,
    contracts         INTEGER NOT NULL DEFAULT 1,
    cost_basis        REAL    NOT NULL DEFAULT 0.0,
    shares            INTEGER NOT NULL DEFAULT 0,
    notes             TEXT    NOT NULL DEFAULT '',
    is_active         INTEGER NOT NULL DEFAULT 1,  -- 1 = open, 0 = closed
    created_at        TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS trade_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL DEFAULT (datetime('now')),
    action        TEXT    NOT NULL,   -- SELL_PUT | ASSIGNED | PUT_EXPIRED_OTM | SELL_CALL | ...
    ticker        TEXT    NOT NULL,
    strike        REAL,
    expiration    TEXT,
    premium       REAL,
    contracts     INTEGER,
    total_premium REAL,
    pnl           REAL,
    notes         TEXT    NOT NULL DEFAULT ''
);

-- Index for fast active-position lookups
CREATE INDEX IF NOT EXISTS idx_positions_active_ticker
    ON positions (ticker, is_active);
"""


def init_db() -> None:
    """Create tables if they don't exist and seed the portfolio_state row."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # executescript auto-commits, so use a raw connection (not our managed one)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.executescript(_SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO portfolio_state (id, cash) VALUES (1, ?)",
        (STARTING_CAPITAL,),
    )
    conn.commit()
    conn.close()
    logger.info("Database initialised at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Portfolio CRUD
# ---------------------------------------------------------------------------

def load_portfolio() -> Portfolio:
    """Load the current portfolio state from SQLite.

    Returns a fresh Portfolio with STARTING_CAPITAL if the DB is empty.
    """
    init_db()  # idempotent — safe to call every time

    with get_connection() as conn:
        # Load scalar state
        row = conn.execute(
            "SELECT cash, total_premium_collected, total_realized_pnl FROM portfolio_state WHERE id = 1"
        ).fetchone()

        portfolio = Portfolio(
            cash=row["cash"],
            total_premium_collected=row["total_premium_collected"],
            total_realized_pnl=row["total_realized_pnl"],
        )

        # Load active positions
        rows = conn.execute(
            """
            SELECT ticker, phase, entry_date, strike, expiration,
                   premium_collected, contracts, cost_basis, shares, notes
            FROM positions
            WHERE is_active = 1
            ORDER BY id
            """
        ).fetchall()

        for r in rows:
            portfolio.positions.append(
                Position(
                    ticker=r["ticker"],
                    phase=r["phase"],
                    entry_date=r["entry_date"],
                    strike=r["strike"],
                    expiration=r["expiration"],
                    premium_collected=r["premium_collected"],
                    contracts=r["contracts"],
                    cost_basis=r["cost_basis"],
                    shares=r["shares"],
                    notes=r["notes"],
                )
            )

    logger.debug(
        "Loaded portfolio: cash=%.2f, positions=%d", portfolio.cash, len(portfolio.positions)
    )
    return portfolio


def save_portfolio(portfolio: Portfolio) -> None:
    """Persist the entire portfolio state atomically.

    Updates the singleton portfolio_state row AND syncs the positions table
    in a single transaction — no partial writes possible.
    """
    with get_connection() as conn:
        # 1. Update scalar totals
        conn.execute(
            """
            UPDATE portfolio_state
            SET cash = ?,
                total_premium_collected = ?,
                total_realized_pnl = ?,
                updated_at = datetime('now')
            WHERE id = 1
            """,
            (
                portfolio.cash,
                portfolio.total_premium_collected,
                portfolio.total_realized_pnl,
            ),
        )

        # 2. Mark ALL existing positions as inactive, then re-insert active ones.
        #    This keeps the audit trail intact (old rows remain with is_active=0).
        conn.execute("UPDATE positions SET is_active = 0 WHERE is_active = 1")

        for p in portfolio.positions:
            # Try to reactivate an exact match first (idempotent re-save)
            updated = conn.execute(
                """
                UPDATE positions
                SET phase = ?,
                    premium_collected = ?,
                    contracts = ?,
                    cost_basis = ?,
                    shares = ?,
                    notes = ?,
                    is_active = 1,
                    updated_at = datetime('now')
                WHERE ticker = ?
                  AND entry_date = ?
                  AND strike = ?
                  AND expiration = ?
                """,
                (
                    p.phase,
                    p.premium_collected,
                    p.contracts,
                    p.cost_basis,
                    p.shares,
                    p.notes,
                    p.ticker,
                    p.entry_date,
                    p.strike,
                    p.expiration,
                ),
            ).rowcount

            if updated == 0:
                # Brand new position — insert
                conn.execute(
                    """
                    INSERT INTO positions
                        (ticker, phase, entry_date, strike, expiration,
                         premium_collected, contracts, cost_basis, shares, notes, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        p.ticker,
                        p.phase,
                        p.entry_date,
                        p.strike,
                        p.expiration,
                        p.premium_collected,
                        p.contracts,
                        p.cost_basis,
                        p.shares,
                        p.notes,
                    ),
                )

    logger.debug(
        "Saved portfolio: cash=%.2f, positions=%d", portfolio.cash, len(portfolio.positions)
    )


# ---------------------------------------------------------------------------
# Trade log
# ---------------------------------------------------------------------------

def record_trade(
    action: str,
    ticker: str,
    strike: float = None,
    expiration: str = None,
    premium: float = None,
    contracts: int = None,
    total_premium: float = None,
    pnl: float = None,
    notes: str = "",
) -> int:
    """Append a trade event to the immutable trade_log table.

    Returns the new row id.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO trade_log
                (action, ticker, strike, expiration, premium, contracts, total_premium, pnl, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (action, ticker, strike, expiration, premium, contracts, total_premium, pnl, notes),
        )
        row_id = cursor.lastrowid

    logger.info("Trade logged: [%s] %s  row_id=%d", action, ticker, row_id)
    return row_id


def get_trade_history(ticker: str = None, limit: int = 100) -> list[dict]:
    """Return trade history rows as plain dicts, newest first.

    Args:
        ticker: Filter by ticker symbol. None = all tickers.
        limit:  Max number of rows to return.
    """
    with get_connection() as conn:
        if ticker:
            rows = conn.execute(
                """
                SELECT * FROM trade_log
                WHERE ticker = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (ticker, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM trade_log
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    return [dict(r) for r in rows]


def get_closed_positions(ticker: str = None) -> list[dict]:
    """Return all historically closed positions."""
    with get_connection() as conn:
        if ticker:
            rows = conn.execute(
                "SELECT * FROM positions WHERE is_active = 0 AND ticker = ? ORDER BY id DESC",
                (ticker,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM positions WHERE is_active = 0 ORDER BY id DESC"
            ).fetchall()
    return [dict(r) for r in rows]
