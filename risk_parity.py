"""risk_parity.py — Risk Parity engine with return targeting.

Algorithm
---------
1. For each rebalance date, build a candidate universe (tickers with enough
   history + CASH).
2. Compute rolling covariance matrix and EWM expected returns.
3. Solve Equal Risk Contribution (ERC) weights via scipy optimisation.
4. Compute max-Sharpe weights via analytical approximation.
5. Blend ERC and max-Sharpe by RETURN_TILT parameter.
6. Select top MAX_ASSETS by blended weight, re-normalise.
7. Store daily portfolio returns and weights.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config

# ── Helpers ───────────────────────────────────────────────────────────────────


def _erc_weights(cov: np.ndarray) -> np.ndarray:
    """Equal Risk Contribution weights via sequential quadratic programming."""
    n = cov.shape[0]
    w0 = np.ones(n) / n

    def _risk_contributions(w: np.ndarray) -> np.ndarray:
        port_var = w @ cov @ w
        mrc = cov @ w  # marginal risk contribution
        return w * mrc / (port_var + 1e-12)

    def _objective(w: np.ndarray) -> float:
        rc = _risk_contributions(w)
        target = 1.0 / n
        return float(np.sum((rc - target) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(config.MIN_WEIGHT, config.MAX_WEIGHT)] * n

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            _objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )

    w = result.x if result.success else w0
    w = np.clip(w, config.MIN_WEIGHT, config.MAX_WEIGHT)
    return w / w.sum()


def _max_sharpe_weights(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """Approximate max-Sharpe weights via mean-variance optimisation."""
    n = len(mu)
    w0 = np.ones(n) / n
    excess = mu - rf

    def _neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ excess)
        port_vol = float(np.sqrt(w @ cov @ w + 1e-12))
        return -port_ret / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(config.MIN_WEIGHT, config.MAX_WEIGHT)] * n

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            _neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500},
        )

    w = result.x if result.success else w0
    w = np.clip(w, config.MIN_WEIGHT, config.MAX_WEIGHT)
    return w / w.sum()


def _select_top_assets(
    weights: np.ndarray, tickers: list[str], max_assets: int
) -> tuple[np.ndarray, list[str]]:
    """Keep top MAX_ASSETS by weight, re-normalise."""
    idx = np.argsort(weights)[::-1][:max_assets]
    idx = np.sort(idx)
    w = weights[idx]
    w = w / w.sum()
    return w, [tickers[i] for i in idx]


# ── Main engine ───────────────────────────────────────────────────────────────


def run_risk_parity(
    log_returns: pd.DataFrame,
    cash_returns: pd.Series,
    universe_tickers: list[str],
    universe_name: str,
) -> dict:
    """Run the risk parity engine for one universe.

    Returns a dict with:
        weights_df   : DataFrame (dates × assets) of daily target weights
        portfolio_ret: Series of daily portfolio log returns
        metadata     : dict of per-rebalance diagnostics
    """
    # Build combined returns (ETFs available in universe + CASH)
    avail = [t for t in universe_tickers if t in log_returns.columns]
    all_assets = avail + ["CASH"]

    rets = pd.concat([log_returns[avail], cash_returns.rename("CASH")], axis=1).dropna()

    dates = rets.index
    T = len(dates)

    print(
        f"\n{'='*60}\n"
        f"Universe: {universe_name}  ({len(avail)} ETFs + CASH)\n"
        f"Period: {dates[0].date()} → {dates[-1].date()}  ({T} days)\n"
        f"{'='*60}"
    )

    # Output containers
    weights_records: list[dict] = []
    port_returns: list[float] = []
    metadata: list[dict] = []

    current_weights: dict[str, float] = {a: 1.0 / len(all_assets) for a in all_assets}
    last_rebal = -config.REBAL_FREQ  # force rebalance on first valid date

    for i, date in enumerate(dates):
        if i < config.COV_WINDOW:
            # Not enough history — hold equal weight
            port_ret = float(
                sum(current_weights.get(a, 0.0) * rets.loc[date, a] for a in all_assets)
            )
            port_returns.append(port_ret)
            weights_records.append({"date": date, **current_weights})
            continue

        # Rebalance on schedule
        if i - last_rebal >= config.REBAL_FREQ:
            window = rets.iloc[i - config.COV_WINDOW : i]

            # Covariance matrix (annualised)
            cov = window.cov().values * 252

            # EWM expected returns (annualised)
            mu = (
                window.ewm(span=config.EWM_SPAN, min_periods=21).mean().iloc[-1].values
                * 252
            )

            # CASH rf for Sharpe (annualised)
            rf_annual = float(cash_returns.loc[:date].iloc[-1] * 252)

            # ERC weights
            w_erc = _erc_weights(cov)

            # Max-Sharpe weights
            w_ms = _max_sharpe_weights(mu, cov, rf=rf_annual)

            # Blend
            tilt = config.RETURN_TILT
            w_blend = (1 - tilt) * w_erc + tilt * w_ms
            w_blend = np.clip(w_blend, config.MIN_WEIGHT, config.MAX_WEIGHT)
            w_blend /= w_blend.sum()

            # Select top MAX_ASSETS
            w_top, top_assets = _select_top_assets(
                w_blend, all_assets, config.MAX_ASSETS
            )

            current_weights = dict(zip(top_assets, w_top))

            # Risk contributions
            top_idx = [all_assets.index(a) for a in top_assets]
            cov_top = cov[np.ix_(top_idx, top_idx)]
            port_vol = float(np.sqrt(w_top @ cov_top @ w_top))
            mrc = cov_top @ w_top
            rc = w_top * mrc / (port_vol**2 + 1e-12)

            port_ret_exp = float(w_top @ mu[top_idx])
            sortino_num = port_ret_exp - rf_annual
            downside = window[top_assets].values @ w_top
            downside_std = float(np.std(downside[downside < 0])) * np.sqrt(252)
            sortino = sortino_num / (downside_std + 1e-8)

            metadata.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "assets": top_assets,
                    "weights": w_top.tolist(),
                    "exp_return_annual": round(port_ret_exp, 6),
                    "port_vol_annual": round(port_vol, 6),
                    "sortino": round(sortino, 4),
                    "risk_contributions": rc.tolist(),
                }
            )

            last_rebal = i
            print(
                f"  {date.date()}  assets={top_assets}  "
                f"exp_ret={port_ret_exp*100:.1f}%  vol={port_vol*100:.1f}%  "
                f"sortino={sortino:.2f}"
            )

        # Daily portfolio return
        port_ret = float(
            sum(current_weights.get(a, 0.0) * rets.loc[date, a] for a in all_assets)
        )
        port_returns.append(port_ret)
        weights_records.append(
            {"date": date, **{a: current_weights.get(a, 0.0) for a in all_assets}}
        )

    weights_df = pd.DataFrame(weights_records).set_index("date").fillna(0.0)
    port_series = pd.Series(port_returns, index=dates, name="portfolio")

    # Summary stats
    ann_ret = float(port_series.mean() * 252)
    ann_vol = float(port_series.std() * np.sqrt(252))
    downside = port_series[port_series < 0]
    ds_std = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else 1e-8
    sortino = ann_ret / ds_std
    cumulative = float(np.expm1(port_series.sum()))
    mdd = float(
        (
            (np.exp(port_series.cumsum()) - np.exp(port_series.cumsum()).cummax())
            / np.exp(port_series.cumsum()).cummax()
        ).min()
    )

    print(
        f"\n  Summary → AnnRet={ann_ret*100:.2f}%  AnnVol={ann_vol*100:.2f}%  "
        f"Sortino={sortino:.2f}  MDD={mdd*100:.1f}%  Cumulative={cumulative*100:.1f}%"
    )

    return {
        "weights_df": weights_df,
        "portfolio_returns": port_series,
        "rebalances": metadata,
        "summary": {
            "ann_return": round(ann_ret, 6),
            "ann_vol": round(ann_vol, 6),
            "sortino": round(sortino, 4),
            "max_drawdown": round(mdd, 6),
            "cumulative_return": round(cumulative, 6),
            "n_rebalances": len(metadata),
            "universe": universe_name,
        },
    }
