import streamlit as st
st.set_page_config(page_title="NZ Power Pulse — Spike Risk", page_icon="⚡", layout="wide")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import app_config as cfg

st.markdown(cfg.CUSTOM_CSS, unsafe_allow_html=True)

st.title("Next-24h Spike Risk")
st.caption(
    "Probability that the spot price exceeds the per-node 95th-percentile "
    "threshold within the next 24 hours. "
    "Synthetic probabilities — will be replaced by classifier output."
)

# --- Sidebar ---
view = st.sidebar.radio("View", ["Single node", "All nodes"])
thresholds = cfg.get_spike_thresholds()


def build_heatmap_data(poc: str) -> tuple:
    spike_df = cfg.generate_synthetic_spike_probs(poc)
    spike_df["date"] = spike_df["timestamp"].dt.date
    spike_df["hour"] = spike_df["timestamp"].dt.hour
    hourly = (spike_df.groupby(["date", "hour"], as_index=False)["spike_prob"]
              .max())
    pivot = hourly.pivot(index="hour", columns="date", values="spike_prob")
    return pivot


def heatmap_trace(pivot, name: str, showscale: bool = True) -> go.Heatmap:
    return go.Heatmap(
        z=pivot.values,
        x=[str(d) for d in pivot.columns],
        y=pivot.index,
        colorscale=cfg.SPIKE_COLORSCALE,
        zmin=0, zmax=1,
        showscale=showscale,
        colorbar=dict(title="P(spike)") if showscale else None,
        name=name,
        hovertemplate="Date: %{x}<br>Hour: %{y}<br>P(spike): %{z:.2f}<extra></extra>",
    )


if view == "Single node":
    poc = st.sidebar.selectbox(
        "Grid node", cfg.POCS,
        format_func=lambda p: f"{p} — {cfg.POC_LABELS[p]}",
    )
    st.metric(
        f"Spike threshold ({poc})",
        f"${thresholds[poc]:.0f}/MWh",
    )

    pivot = build_heatmap_data(poc)
    fig = go.Figure(data=heatmap_trace(pivot, poc))
    fig.update_layout(
        yaxis_title="Hour of day (UTC)",
        xaxis_title="Date",
        height=400,
        yaxis=dict(dtick=2),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"spike_{poc}")

    with st.expander("Spike statistics"):
        spike_df = cfg.generate_synthetic_spike_probs(poc)
        total = len(spike_df)
        spikes = spike_df["actual_spike"].sum()
        st.write(f"Total half-hours: {total:,}")
        st.write(f"Actual spikes (next 24h): {int(spikes):,} ({spikes/total*100:.1f}%)")
        st.write(f"Mean predicted probability: {spike_df['spike_prob'].mean():.3f}")

else:
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[f"{p} — {cfg.POC_PROFILES[p]}" for p in cfg.POCS],
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
    )

    for i, poc in enumerate(cfg.POCS):
        row, col = (i // 2) + 1, (i % 2) + 1
        pivot = build_heatmap_data(poc)
        trace = heatmap_trace(pivot, poc, showscale=(i == 0))
        fig.add_trace(trace, row=row, col=col)
        fig.update_yaxes(dtick=4, row=row, col=col)

    fig.update_layout(height=900)
    st.plotly_chart(fig, use_container_width=True, key="spike_all")

    st.subheader("Spike Thresholds (95th percentile, training data)")
    cols = st.columns(3)
    for i, poc in enumerate(cfg.POCS):
        with cols[i % 3]:
            st.metric(
                f"{poc}",
                f"${thresholds[poc]:.0f}/MWh",
            )
