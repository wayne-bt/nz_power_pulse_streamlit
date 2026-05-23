import streamlit as st
st.set_page_config(
    page_title="NZ Power Pulse",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

import app_config as cfg

st.markdown(cfg.CUSTOM_CSS, unsafe_allow_html=True)

st.title("NZ Power Pulse")
st.markdown(
    "Half-hourly wholesale electricity spot-price forecasting "
    "across 6 grid nodes, benchmarking classical ML against "
    "Chronos-2 foundation time-series models."
)

# --- Key stats ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Grid Nodes", "6")
col2.metric("Models", str(len(cfg.MODEL_KEYS)))
col3.metric("Time Span", "3.9 years")
col4.metric("Panel Rows", "412K")

st.divider()

# --- Leaderboard ---
st.subheader("Model Leaderboard")

leaderboard = cfg.build_leaderboard()

summary = (
    leaderboard
    .groupby("Model", sort=False)
    .agg({"RMSE ($/MWh)": "mean", "sMAPE (%)": "mean"})
    .round({"RMSE ($/MWh)": 1, "sMAPE (%)": 1})
    .reset_index()
)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "RMSE ($/MWh)": st.column_config.NumberColumn(format="%.1f"),
        "sMAPE (%)": st.column_config.NumberColumn(format="%.1f%%"),
    },
)

with st.expander("Per-node breakdown"):
    pivot = leaderboard.pivot_table(
        index="Model", columns="Node", values="RMSE ($/MWh)",
    ).round(1)
    st.dataframe(pivot, use_container_width=True)

st.divider()

# --- Node overview ---
st.subheader("Grid Nodes")
cols = st.columns(3)
for i, poc in enumerate(cfg.POCS):
    with cols[i % 3]:
        color = cfg.NODE_COLORS[poc]
        st.markdown(
            f"<span style='color:{color};font-weight:bold'>{poc}</span> "
            f"&mdash; {cfg.POC_LABELS[poc]}<br>"
            f"<small style='color:{cfg.COLORS['text_muted']}'>"
            f"{cfg.POC_PROFILES[poc]}</small>",
            unsafe_allow_html=True,
        )

st.divider()
st.caption(
    "Navigate using the sidebar: "
    "**Model Comparison** (forecast accuracy by node) · "
    "**Spike Risk** (next-24h spike heatmap) · "
    "**Dry Year Story** (the 2024 crisis narrative) · "
    "**Predict** (interactive price prediction)"
)
