"""app.py — Risk Parity Engine Dashboard."""

from __future__ import annotations

import os
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import config
from us_calendar import next_trading_day

st.set_page_config(
    page_title="Risk Parity · P2Quant",
    layout="wide",
    page_icon="⚖️",
)

HF_TOKEN = os.environ.get("HF_TOKEN")
BASE_RAW = f"https://huggingface.co/datasets/{config.HF_OUTPUT_REPO}/resolve/main"
BASE_API = f"https://huggingface.co/api/datasets/{config.HF_OUTPUT_REPO}/tree/main"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

UNIVERSE_ICONS = {"EQUITY_SECTORS": "📊", "COMBINED": "🌐"}
COLOURS = [
    "#1B4F8A",
    "#27AE60",
    "#E74C3C",
    "#F39C12",
    "#8E44AD",
    "#148F77",
    "#CA6F1E",
    "#2471A3",
    "#717D7E",
    "#B7950B",
]


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Loading results…")
def load_latest_json(universe: str) -> dict | None:
    slug = universe.lower().replace("_", "-")
    try:
        r = requests.get(BASE_API, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return None
        files = sorted([f["path"] for f in r.json() if f["path"].endswith(".json")])
        # Latest file for this universe
        matches = [f for f in files if f"_{slug}.json" in f]
        if not matches:
            return None
        url = f"{BASE_RAW}/{matches[-1]}"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner="Loading portfolio returns…")
def load_portfolio_returns(universe: str) -> pd.DataFrame | None:
    slug = universe.lower().replace("_", "-")
    url = f"{BASE_RAW}/portfolio_returns_{slug}.csv"
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        if r.status_code != 200:
            return None
        df = pd.read_csv(StringIO(r.text), parse_dates=["date"])
        return df.set_index("date")
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner="Loading weights history…")
def load_weights(universe: str) -> pd.DataFrame | None:
    slug = universe.lower().replace("_", "-")
    url = f"{BASE_RAW}/weights_{slug}.csv"
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        if r.status_code != 200:
            return None
        df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
        return df
    except Exception:
        return None


def fmt_pct(v: float) -> str:
    return f"{v * 100:+.2f}%"


def colour_for(asset: str, i: int) -> str:
    return COLOURS[i % len(COLOURS)]


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚖️ Risk Parity Engine")
st.caption(
    "Equal Risk Contribution + Return Tilt · Max 8 assets (ETFs + CASH) · "
    "Monthly rebalance · Sortino-optimised"
)

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    universe = st.selectbox(
        "Universe",
        list(config.UNIVERSES.keys()),
        format_func=lambda u: f"{UNIVERSE_ICONS.get(u, '📈')} {u}",
    )
    st.divider()
    st.markdown(f"**Tickers:** {len(config.UNIVERSES[universe])} ETFs + CASH")
    st.markdown(f"**Cov window:** {config.COV_WINDOW} days")
    st.markdown(f"**Rebal freq:** every {config.REBAL_FREQ} days")
    st.markdown(f"**Return tilt:** {config.RETURN_TILT:.0%}")
    st.markdown(f"**Max assets:** {config.MAX_ASSETS}")
    st.markdown(f"**Next trading day:** {next_trading_day()}")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────
data = load_latest_json(universe)
port_rets = load_portfolio_returns(universe)
weights_hist = load_weights(universe)

if data is None:
    st.warning("⚠️ No results found. Run `python trainer.py` first.")
    st.stop()

summary = data.get("summary", {})
latest_weights = data.get("latest_weights", {})
latest_date = data.get("latest_date", "?")
rebalances = data.get("rebalances", [])

# Filter to assets with non-zero weight
latest_weights = {k: v for k, v in latest_weights.items() if v > 0.001}
assets = sorted(latest_weights, key=latest_weights.get, reverse=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
h1, h2, h3 = st.columns(3)
h1.metric("Run Date", data.get("run_date", "?"))
h2.metric("Latest Weights Date", latest_date)
h3.metric("Universe", f"{UNIVERSE_ICONS.get(universe, '')} {universe}")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Ann. Return", fmt_pct(summary.get("ann_return", 0)))
k2.metric("Ann. Volatility", fmt_pct(summary.get("ann_vol", 0)))
k3.metric("Sortino", f"{summary.get('sortino', 0):.2f}")
k4.metric("Max Drawdown", fmt_pct(summary.get("max_drawdown", 0)))
k5.metric("Cumulative", fmt_pct(summary.get("cumulative_return", 0)))

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎯 Current Portfolio",
        "📈 Performance",
        "🗂️ Weight History",
        "📋 Rebalance Log",
    ]
)

with tab1:
    st.subheader(f"Portfolio as of {latest_date}")

    pie_col, table_col = st.columns([1, 1])

    with pie_col:
        fig_pie = go.Figure(
            go.Pie(
                labels=assets,
                values=[latest_weights[a] for a in assets],
                marker=dict(colors=[colour_for(a, i) for i, a in enumerate(assets)]),
                texttemplate="%{label}<br>%{percent}",
                hole=0.35,
            )
        )
        fig_pie.update_layout(
            title=f"Current Allocation — {universe}",
            height=420,
            margin=dict(t=50, b=20, l=20, r=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="pie_current")

    with table_col:
        # Latest rebalance detail
        if rebalances:
            last_reb = rebalances[-1]
            st.markdown(f"**Latest rebalance:** {last_reb['date']}")
            reb_assets = last_reb.get("assets", [])
            reb_weights = last_reb.get("weights", [])
            reb_rc = last_reb.get("risk_contributions", [])

            rows = []
            for a, w, rc in zip(reb_assets, reb_weights, reb_rc):
                rows.append(
                    {
                        "Asset": a,
                        "Weight": f"{w*100:.1f}%",
                        "Risk Contrib.": f"{rc*100:.1f}%",
                        "Δ (W-RC)": f"{(w-rc)*100:+.1f}%",
                    }
                )
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Exp. Annual Return", fmt_pct(last_reb.get("exp_return_annual", 0))
            )
            m2.metric("Port. Volatility", fmt_pct(last_reb.get("port_vol_annual", 0)))
            m3.metric("Sortino", f"{last_reb.get('sortino', 0):.2f}")

        # Horizontal weight bar
        fig_bar = go.Figure(
            go.Bar(
                y=assets,
                x=[latest_weights[a] * 100 for a in assets],
                orientation="h",
                marker_color=[colour_for(a, i) for i, a in enumerate(assets)],
                text=[f"{latest_weights[a]*100:.1f}%" for a in assets],
                textposition="outside",
            )
        )
        fig_bar.update_layout(
            title="Weight by Asset",
            xaxis_title="Weight (%)",
            yaxis=dict(autorange="reversed"),
            height=300,
            margin=dict(t=40, b=20, l=60, r=60),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="bar_weights")

with tab2:
    st.subheader("Portfolio Performance")
    if port_rets is not None and not port_rets.empty:
        cum_ret = np.exp(port_rets["portfolio_return"].cumsum()) - 1

        # Cumulative return chart
        fig_perf = go.Figure()
        fig_perf.add_trace(
            go.Scatter(
                x=cum_ret.index,
                y=cum_ret.values * 100,
                mode="lines",
                name="Risk Parity",
                line=dict(color="#1B4F8A", width=2),
                fill="tozeroy",
                fillcolor="rgba(27,79,138,0.1)",
            )
        )
        fig_perf.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_perf.update_layout(
            title=f"Cumulative Return — {universe}",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_perf, use_container_width=True, key="perf_chart")

        # Rolling Sortino
        daily = port_rets["portfolio_return"]
        roll_window = 63
        roll_mean = daily.rolling(roll_window).mean() * 252
        roll_down = daily.rolling(roll_window).apply(
            lambda x: x[x < 0].std() * np.sqrt(252) if (x < 0).any() else 1e-8
        )
        roll_sortino = roll_mean / (roll_down + 1e-8)

        fig_sort = go.Figure()
        fig_sort.add_trace(
            go.Scatter(
                x=roll_sortino.index,
                y=roll_sortino.values,
                mode="lines",
                name="Rolling Sortino (63d)",
                line=dict(color="#27AE60", width=1.5),
            )
        )
        fig_sort.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_sort.add_hline(
            y=1, line_dash="dash", line_color="#F39C12", annotation_text="Sortino=1"
        )
        fig_sort.update_layout(
            title="Rolling 63-day Sortino Ratio",
            height=280,
            margin=dict(t=40, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_sort, use_container_width=True, key="sortino_chart")

        # Drawdown
        cum = np.exp(daily.cumsum())
        peak = cum.cummax()
        dd = (cum - peak) / peak * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(231,76,60,0.3)",
                line=dict(color="#E74C3C", width=1),
                name="Drawdown",
            )
        )
        fig_dd.update_layout(
            title="Drawdown (%)",
            height=250,
            margin=dict(t=40, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dd, use_container_width=True, key="dd_chart")

        # Monthly returns heatmap
        monthly = daily.resample("ME").sum()
        monthly_df = pd.DataFrame(
            {
                "year": monthly.index.year,
                "month": monthly.index.month,
                "ret": monthly.values * 100,
            }
        )
        pivot = monthly_df.pivot(index="year", columns="month", values="ret")
        pivot.columns = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ][: len(pivot.columns)]

        fig_hm = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="RdYlGn",
                zmid=0,
                colorbar=dict(title="Return %"),
                text=[
                    [f"{v:.1f}%" if not np.isnan(v) else "" for v in row]
                    for row in pivot.values
                ],
                texttemplate="%{text}",
                hoverongaps=False,
            )
        )
        fig_hm.update_layout(
            title="Monthly Returns Heatmap (%)",
            height=max(300, len(pivot) * 28 + 80),
            margin=dict(t=40, b=40, l=60, r=20),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hm, use_container_width=True, key="monthly_hm")
    else:
        st.info("No portfolio return data found.")

with tab3:
    st.subheader("Weight History")
    if weights_hist is not None and not weights_hist.empty:
        # Drop zero columns
        w = weights_hist.loc[:, (weights_hist > 0.001).any()]
        asset_cols = [c for c in w.columns]

        # Stacked area chart
        fig_area = go.Figure()
        for i, col in enumerate(asset_cols):
            fig_area.add_trace(
                go.Scatter(
                    x=w.index,
                    y=w[col] * 100,
                    mode="lines",
                    stackgroup="one",
                    name=col,
                    line=dict(width=0.5, color=colour_for(col, i)),
                    fillcolor=colour_for(col, i),
                )
            )
        fig_area.update_layout(
            title=f"Portfolio Allocation Over Time — {universe}",
            yaxis_title="Weight (%)",
            xaxis_title="Date",
            height=450,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_area, use_container_width=True, key="area_weights")

        # Rebalance-date weight heatmap
        if rebalances:
            reb_dates = [r["date"] for r in rebalances]
            all_reb_assets = sorted(set(a for r in rebalances for a in r["assets"]))
            heat_z = []
            for r in rebalances:
                w_map = dict(zip(r["assets"], r["weights"]))
                heat_z.append([w_map.get(a, 0.0) * 100 for a in all_reb_assets])

            fig_reb_hm = go.Figure(
                go.Heatmap(
                    z=heat_z,
                    x=all_reb_assets,
                    y=reb_dates,
                    colorscale="Blues",
                    colorbar=dict(title="Weight %"),
                    text=[[f"{v:.1f}%" for v in row] for row in heat_z],
                    texttemplate="%{text}",
                    hoverongaps=False,
                )
            )
            fig_reb_hm.update_layout(
                title="Weight at Each Rebalance",
                height=max(300, len(reb_dates) * 18 + 80),
                margin=dict(t=40, b=60, l=80, r=20),
                xaxis=dict(tickangle=-45),
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_reb_hm, use_container_width=True, key="reb_hm")
    else:
        st.info("No weight history found.")

with tab4:
    st.subheader("Rebalance Log")
    if rebalances:
        rows = []
        for r in reversed(rebalances):
            rows.append(
                {
                    "Date": r["date"],
                    "Assets": ", ".join(r["assets"]),
                    "Exp. Return": fmt_pct(r.get("exp_return_annual", 0)),
                    "Volatility": fmt_pct(r.get("port_vol_annual", 0)),
                    "Sortino": f"{r.get('sortino', 0):.2f}",
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            height=500,
        )

        # Sortino over rebalances
        dates_r = [r["date"] for r in rebalances]
        sortinos_r = [r.get("sortino", 0) for r in rebalances]
        fig_rs = go.Figure(
            go.Scatter(
                x=dates_r,
                y=sortinos_r,
                mode="lines+markers",
                line=dict(color="#27AE60", width=2),
                marker=dict(size=5),
                name="Sortino",
            )
        )
        fig_rs.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_rs.update_layout(
            title="Expected Sortino at Each Rebalance",
            yaxis_title="Sortino",
            height=300,
            margin=dict(t=40, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_rs, use_container_width=True, key="reb_sortino")
    else:
        st.info("No rebalance data found.")

st.divider()
st.caption(
    f"P2Quant Risk Parity Engine · Run: {data.get('run_date', '?')} · "
    f"Data: {config.HF_DATA_REPO} · Results: {config.HF_OUTPUT_REPO}"
)
