"""Global configuration for the Options Trading Competition project."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Competition parameters
STARTING_CAPITAL = 1_000_000  # $1M paper trading
COMPETITION_START = "2026-04-01"
COMPETITION_END = "2026-06-30"

# Wheel strategy constraints
SHARES_PER_CONTRACT = 100
MAX_POSITION_PCT = 0.15          # Max 15% of portfolio in one ticker
MIN_CASH_RESERVE_PCT = 0.10      # Keep 10% cash reserve
MAX_CONCURRENT_WHEELS = 8        # Max different tickers at once

# Stock screening criteria
MIN_STOCK_PRICE = 20             # Avoid penny stocks
MAX_STOCK_PRICE = 500            # Keep position sizes manageable (100 shares)
MIN_OPTION_VOLUME = 100          # Minimum daily option volume
MIN_IV_RANK = 30                 # Minimum IV rank (percentile) for good premiums
MIN_MARKET_CAP = 5_000_000_000   # $5B+ market cap for stability

# Options selection defaults
DEFAULT_DTE_MIN = 20             # Minimum days to expiration
DEFAULT_DTE_MAX = 45             # Maximum days to expiration (sweet spot for theta)
DEFAULT_DELTA_PUT = -0.30        # Target delta for cash-secured puts (~70% OTM)
DEFAULT_DELTA_CALL = 0.30        # Target delta for covered calls

# Risk management
MAX_LOSS_PER_POSITION_PCT = 0.05  # Stop-loss at 5% of portfolio per position
EARNINGS_BUFFER_DAYS = 7          # Avoid selling options within 7 days of earnings
