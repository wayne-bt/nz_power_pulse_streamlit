import streamlit as st
st.set_page_config(page_title="NZ Power Pulse — Dry Year Story", page_icon="⚡", layout="wide")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import app_config as cfg

st.markdown(cfg.CUSTOM_CSS, unsafe_allow_html=True)

st.title("The Dry-Year Crisis")

st.markdown("""
Between April and August 2024, NZ hydro storage fell to a six-year low.
Gas supply from Maui/Pohokura was already scarce. Wholesale spot prices
exceeded **NZ$800/MWh** at peak, Methanex idled production, and the Tiwai
smelter cut demand by ~330 GWh. By Q4 2025 hydro recovered and NZ hit a
record **96.4% renewable share** — the largest swing in modern NZ
electricity history.

This page walks through the crisis using the gold panel data across all
six grid nodes.
""")

panel = cfg.load_gold_panel()

# --- Monthly average price by node ---
st.subheader("Monthly Average Spot Price by Node")

monthly = (panel.groupby([panel["ts_utc"].dt.to_period("M"), "POC"])
           ["price_dmwh"].mean().reset_index())
monthly["month"] = monthly["ts_utc"].dt.to_timestamp()

fig_price = go.Figure()
for poc in cfg.POCS:
    sub = monthly[monthly["POC"] == poc]
    fig_price.add_trace(go.Scatter(
        x=sub["month"], y=sub["price_dmwh"],
        name=f"{poc} ({cfg.POC_PROFILES[poc]})",
        line=dict(color=cfg.NODE_COLORS[poc], width=2),
    ))

fig_price.add_vrect(
    x0="2024-04-01", x1="2024-09-01",
    fillcolor=cfg.COLORS["danger"], opacity=0.08,
    line_width=0,
    annotation_text="Dry-year crisis",
    annotation_position="top left",
    annotation_font_color=cfg.COLORS["danger"],
)
fig_price.update_layout(
    yaxis_title="NZ$/MWh (monthly mean)",
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig_price, use_container_width=True, key="monthly_price")

# --- Fuel mix ---
st.subheader("National Generation Mix")
st.markdown(
    "Hydro share collapsed during the crisis as lake levels dropped. "
    "Thermal generation filled the gap — the price signal that drives spot above $300/MWh."
)

fuel_cols = ["hydro_share", "thermal_share", "wind_share", "geo_share"]
fuel_labels = {"hydro_share": "Hydro", "thermal_share": "Thermal",
               "wind_share": "Wind", "geo_share": "Geothermal"}
fuel_colors = {"hydro_share": "#00d4ff", "thermal_share": "#ff4444",
               "wind_share": "#00e676", "geo_share": "#f0b429"}

national = (panel.groupby(panel["ts_utc"].dt.to_period("M"))[fuel_cols]
            .mean().reset_index())
national["month"] = national["ts_utc"].dt.to_timestamp()

fig_fuel = go.Figure()
for col in fuel_cols:
    fig_fuel.add_trace(go.Scatter(
        x=national["month"], y=national[col] * 100,
        name=fuel_labels[col],
        line=dict(color=fuel_colors[col], width=2),
        stackgroup="fuel",
    ))

fig_fuel.add_vrect(
    x0="2024-04-01", x1="2024-09-01",
    fillcolor=cfg.COLORS["danger"], opacity=0.08,
    line_width=0,
)
fig_fuel.update_layout(
    yaxis_title="Share of generation (%)",
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig_fuel, use_container_width=True, key="fuel_mix")

# --- Crisis zoom: daily max price ---
st.subheader("Crisis Period: Daily Peak Prices (Apr–Sep 2024)")

crisis = panel[
    (panel["ts_utc"] >= "2024-04-01") & (panel["ts_utc"] < "2024-10-01")
].copy()
daily_max = (crisis.groupby([crisis["ts_utc"].dt.date, "POC"])
             ["price_dmwh"].max().reset_index())
daily_max.columns = ["date", "POC", "peak_price"]
daily_max["date"] = pd.to_datetime(daily_max["date"])

fig_crisis = go.Figure()
for poc in cfg.POCS:
    sub = daily_max[daily_max["POC"] == poc]
    fig_crisis.add_trace(go.Scatter(
        x=sub["date"], y=sub["peak_price"],
        name=f"{poc}",
        line=dict(color=cfg.NODE_COLORS[poc], width=1.5),
    ))

fig_crisis.add_hline(
    y=300, line_dash="dash", line_color=cfg.COLORS["danger"],
    annotation_text="$300 stress line",
    annotation_font_color=cfg.COLORS["danger"],
)
fig_crisis.update_layout(
    yaxis_title="Daily peak NZ$/MWh",
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig_crisis, use_container_width=True, key="crisis_zoom")

st.markdown("""
**Key observations:**

- **South Island hydro nodes** (Benmore, Roxburgh) spiked first — they sit
  closest to the depleted lakes and act as the system's price-discovery points.
- **North Island load nodes** (Otahuhu, Haywards) followed within hours as the
  HVDC link transmitted the scarcity signal northward.
- **Kawerau** (geothermal) was the most stable — geothermal baseload doesn't
  depend on rainfall and acts as a natural price anchor.
- **Whirinaki** (thermal peaker) spiked on days when it was dispatched as
  last-resort generation — a direct dry-year stress indicator.
""")

# --- Load vs Gen node comparison ---
st.subheader("Load vs Generation Nodes: Crisis Response")

crisis_monthly = (
    crisis.groupby([crisis["ts_utc"].dt.to_period("M"), "node_type"])
    ["price_dmwh"].agg(["mean", "max"]).reset_index()
)
crisis_monthly["month"] = crisis_monthly["ts_utc"].dt.to_timestamp()

fig_lg = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Mean Price", "Peak Price"],
)
for ntype, color in [("load", cfg.COLORS["primary"]), ("gen", cfg.COLORS["secondary"])]:
    sub = crisis_monthly[crisis_monthly["node_type"] == ntype]
    fig_lg.add_trace(
        go.Bar(x=sub["month"], y=sub["mean"], name=f"{ntype} (mean)",
               marker_color=color, opacity=0.8),
        row=1, col=1,
    )
    fig_lg.add_trace(
        go.Bar(x=sub["month"], y=sub["max"], name=f"{ntype} (peak)",
               marker_color=color, opacity=0.8, showlegend=False),
        row=1, col=2,
    )

fig_lg.update_layout(
    height=350, barmode="group",
    legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
)
fig_lg.update_yaxes(title_text="NZ$/MWh", row=1, col=1)
fig_lg.update_yaxes(title_text="NZ$/MWh", row=1, col=2)
st.plotly_chart(fig_lg, use_container_width=True, key="load_vs_gen")

st.caption(
    "Data source: EMI wholesale spot prices + Open-Meteo ERA5 weather reanalysis. "
    "All timestamps in UTC."
)
