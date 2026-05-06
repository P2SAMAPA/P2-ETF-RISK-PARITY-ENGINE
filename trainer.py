"""trainer.py — Risk Parity engine orchestrator."""

from __future__ import annotations

import io
import json
import os

from huggingface_hub import HfApi

import config
import data_manager
from risk_parity import run_risk_parity


def push_results(result: dict, universe: str, token: str) -> None:
    slug = universe.lower().replace("_", "-")
    filename = f"rp_{config.TODAY}_{slug}.json"
    api = HfApi(token=token)
    api.create_repo(
        repo_id=config.HF_OUTPUT_REPO,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )

    # Build JSON-serialisable output (exclude DataFrames)
    output = {
        "run_date": config.TODAY,
        "universe": universe,
        "summary": result["summary"],
        "rebalances": result["rebalances"],
        "latest_weights": result["weights_df"].iloc[-1].to_dict(),
        "latest_date": result["weights_df"].index[-1].strftime("%Y-%m-%d"),
    }

    # Also push the portfolio returns as CSV
    ret_csv = result["portfolio_returns"].reset_index()
    ret_csv.columns = ["date", "portfolio_return"]
    ret_csv["date"] = ret_csv["date"].dt.strftime("%Y-%m-%d")
    csv_bytes = ret_csv.to_csv(index=False).encode()
    api.upload_file(
        path_or_fileobj=io.BytesIO(csv_bytes),
        path_in_repo=f"portfolio_returns_{slug}.csv",
        repo_id=config.HF_OUTPUT_REPO,
        repo_type="dataset",
        commit_message=f"Portfolio returns {config.TODAY} — {slug}",
    )

    # Push weights CSV
    weights_csv = result["weights_df"].copy()
    weights_csv.index = weights_csv.index.strftime("%Y-%m-%d")
    weights_bytes = weights_csv.to_csv().encode()
    api.upload_file(
        path_or_fileobj=io.BytesIO(weights_bytes),
        path_in_repo=f"weights_{slug}.csv",
        repo_id=config.HF_OUTPUT_REPO,
        repo_type="dataset",
        commit_message=f"Weights {config.TODAY} — {slug}",
    )

    # Push summary JSON
    json_bytes = json.dumps(output, indent=2, default=str).encode()
    api.upload_file(
        path_or_fileobj=io.BytesIO(json_bytes),
        path_in_repo=filename,
        repo_id=config.HF_OUTPUT_REPO,
        repo_type="dataset",
        commit_message=f"Risk Parity results {config.TODAY} — {slug}",
    )
    print(f"Pushed → {config.HF_OUTPUT_REPO}/{filename}")


def main() -> None:
    token = config.HF_TOKEN
    if not token:
        print("HF_TOKEN not set — aborting.")
        return

    target = os.environ.get("RP_UNIVERSE", "ALL").upper()
    log_returns, cash_returns = data_manager.load_data(token=token)

    for universe_name, tickers in config.UNIVERSES.items():
        if target != "ALL" and universe_name != target:
            continue

        result = run_risk_parity(
            log_returns=log_returns,
            cash_returns=cash_returns,
            universe_tickers=tickers,
            universe_name=universe_name,
        )
        push_results(result, universe_name, token)

    print("\n✅ Risk Parity engine complete.")


if __name__ == "__main__":
    main()
