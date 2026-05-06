"""data_manager.py — Data loading for Risk Parity engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

import config

ETF_TICKERS = sorted(set(config.EQUITY_SECTORS_TICKERS + config.FI_COMMODITIES_TICKERS))


def load_data(token: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load master data.

    Returns
    -------
    log_returns : DataFrame, tickers as columns
    cash_returns : Series, daily cash (T-bill) log returns
    """
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=token,
        cache_dir="./hf_cache",
    )
    df = pd.read_parquet(file_path)

    # Normalise index
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.set_index("Date")

    # ── ETF log returns ───────────────────────────────────────────────────────
    available = [t for t in ETF_TICKERS if t in df.columns]
    prices = df[available].ffill()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # ── CASH: 3M T-Bill annualised → daily simple → log ──────────────────────
    if "TBILL_3M" in df.columns:
        tbill = df["TBILL_3M"].reindex(log_returns.index).ffill().fillna(0.0)
        # Convert annual % to daily log return: ln(1 + r_annual/100)^(1/252)
        cash_log = np.log1p(tbill / 100.0) / 252.0
    else:
        cash_log = pd.Series(0.0, index=log_returns.index, name="CASH")

    cash_log.name = "CASH"

    print(
        f"Loaded {len(log_returns)} rows × {len(log_returns.columns)} ETFs "
        f"| CASH mean daily={cash_log.mean()*100:.4f}%"
    )
    return log_returns, cash_log
