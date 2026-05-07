"""config.py — Risk Parity Engine configuration."""

import os
from datetime import datetime

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-risk-parity-results"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# ── Universes ─────────────────────────────────────────────────────────────────
EQUITY_SECTORS_TICKERS = [
    "SPY",
    "QQQ",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "GDX",
    "XME",
    "IWF",
    "XSD",
    "XBI",
    "IWM",
]
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
COMBINED_TICKERS = sorted(set(EQUITY_SECTORS_TICKERS + FI_COMMODITIES_TICKERS))

# CASH is always added to each universe
UNIVERSES = {
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": COMBINED_TICKERS,
}

# ── Risk Parity Parameters ────────────────────────────────────────────────────
COV_WINDOW = 63  # Rolling covariance window (days) — ~1 quarter
EWM_SPAN = 63  # EWM span for expected return estimation
REBAL_FREQ = 21  # Rebalance every N trading days (~monthly)
MAX_ASSETS = 8  # Max ETFs (+ CASH) in output portfolio
MIN_WEIGHT = 0.01  # Minimum weight per asset (1%)
MAX_WEIGHT = 0.40  # Maximum weight per asset (40%)
RISK_FREE_ANNUAL = 0.0  # Annualised risk-free for Sharpe (0 = use CASH)

# Return targeting: blend ERC weights with max-Sharpe tilt
# 0.0 = pure ERC, 1.0 = pure max-Sharpe
RETURN_TILT = 0.72

# ── Data ──────────────────────────────────────────────────────────────────────
TRAIN_START = "2008-01-01"
TODAY = datetime.now().strftime("%Y-%m-%d")
