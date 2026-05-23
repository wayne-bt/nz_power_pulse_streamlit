import streamlit as st
st.set_page_config(page_title="NZ Power Pulse — Forecast", page_icon="⚡", layout="wide")

import plotly.graph_objects as go
import pandas as pd
import app_config as cfg

st.markdown(cfg.CUSTOM_CSS, unsafe_allow_html=True)

st.title("24-Hour Price Forecast")

# --- Sidebar controls ---
poc = st.sidebar.selectbox(
    "Grid node",
    cfg.POCS,
    format_func=lambda p: f"{p} — {cfg.POC_LABELS[p]}",
)

selected_models = st.sidebar.multiselect(
    "Models",
    cfg.MODEL_KEYS,
    default=["catboost_regressor", "chronos2s_finetune"],
    format_func=lambda m: cfg.MODEL_LABELS[m],
)

show_fan = st.sidebar.checkbox("Show P10/P90 prediction interval", value=True)

# --- Load data ---
datasets = {}
for model in selected_models:
    df = cfg.load_predictions(model, poc)
    if not df.empty:
        datasets[model] = df

if not datasets:
    st.warning("No prediction data found for selected models and node.")
    st.stop()

# --- Date range from available data ---
all_ts = pd.concat([d["timestamp"] for d in datasets.values()])
ts_min, ts_max = all_ts.min(), all_ts.max()

# Key resets the widget when model selection changes, avoiding stale-value errors
range_key = f"dr_{'_'.join(sorted(datasets.keys()))}"
date_range = st.sidebar.date_input(
    "Date range",
    value=(ts_min.date(), ts_max.date()),
    min_value=ts_min.date(),
    max_value=ts_max.date(),
    key=range_key,
)
if len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
else:
    start, end = ts_min, ts_max + pd.Timedelta(days=1)

# --- Build chart ---
fig = go.Figure()

first_model = selected_models[0] if selected_models else None
first_df = datasets.get(first_model, pd.DataFrame())
if not first_df.empty:
    mask = (first_df["timestamp"] >= start) & (first_df["timestamp"] < end)
    actual_data = first_df[mask].sort_values("timestamp")
    fig.add_trace(go.Scatter(
        x=actual_data["timestamp"],
        y=actual_data["actual"],
        name="Actual",
        line=dict(color=cfg.COLORS["text"], width=1.5),
        opacity=0.7,
    ))

for model in selected_models:
    df = datasets.get(model, pd.DataFrame())
    if df.empty:
        continue
    mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
    sub = df[mask].sort_values("timestamp")
    color = cfg.MODEL_COLORS[model]

    fig.add_trace(go.Scatter(
        x=sub["timestamp"],
        y=sub["predicted"],
        name=cfg.MODEL_LABELS[model],
        line=dict(color=color, width=2),
    ))

    has_quantiles = show_fan and sub["p10"].notna().any()
    if has_quantiles and len(selected_models) == 1:
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fill_rgba = f"rgba({r},{g},{b},0.12)"
        fig.add_trace(go.Scatter(
            x=sub["timestamp"], y=sub["p90"],
            mode="lines", line=dict(width=0), showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=sub["timestamp"], y=sub["p10"],
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor=fill_rgba,
            name="P10–P90",
            hoverinfo="skip",
        ))

fig.update_layout(
    yaxis_title="NZ$/MWh",
    xaxis_title="Time (UTC)",
    height=500,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig, use_container_width=True, key="forecast_chart")

# --- Metrics ---
st.subheader("Model Metrics")
metric_cols = st.columns(len(selected_models))
for i, model in enumerate(selected_models):
    m = cfg.compute_metrics(model, poc)
    with metric_cols[i]:
        st.markdown(
            f"<small style='color:{cfg.MODEL_COLORS[model]}'>"
            f"{cfg.MODEL_LABELS[model]}</small>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        c1.metric("RMSE", f"${m['rmse']:.1f}")
        c2.metric("sMAPE", f"{m['smape']:.1f}%")

# --- Data coverage note ---
with st.expander("Data coverage"):
    for model in selected_models:
        df = datasets.get(model, pd.DataFrame())
        if df.empty:
            st.write(f"**{cfg.MODEL_LABELS[model]}**: no data")
        else:
            st.write(
                f"**{cfg.MODEL_LABELS[model]}**: "
                f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()} "
                f"({len(df):,} rows, horizons {df['horizon_h'].min()}–{df['horizon_h'].max()})"
            )
