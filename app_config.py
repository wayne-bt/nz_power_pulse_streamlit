from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

BASE_DIR = Path(__file__).parent
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
GOLD_PANEL_PATH = BASE_DIR / "data" / "gold" / "panel.parquet"
FEATURES_PATH = BASE_DIR / "data" / "processed" / "df_features.parquet"

POCS = ["OTA2201", "HAY2201", "BEN2201", "ROX1101", "KAW1101", "WHI2201"]

GOLD_POC_MAP = {
    "OTA0221": "OTA2201",
    "HAY0331": "HAY2201",
    "BEN2202": "BEN2201",
}

POC_LABELS = {
    "OTA2201": "Otahuhu (Auckland)",
    "HAY2201": "Haywards (Wellington)",
    "BEN2201": "Benmore (Canterbury)",
    "ROX1101": "Roxburgh (Otago)",
    "KAW1101": "Kawerau (Bay of Plenty)",
    "WHI2201": "Whirinaki (Hawke's Bay)",
}

POC_PROFILES = {
    "OTA2201": "North Island — load centre",
    "HAY2201": "North Island — load centre",
    "BEN2201": "South Island — hydro hub",
    "ROX1101": "South Island — hydro",
    "KAW1101": "North Island — geothermal",
    "WHI2201": "North Island — thermal peaker",
}

NODE_COLORS = {
    "OTA2201": "#f0b429",
    "HAY2201": "#00d4ff",
    "BEN2201": "#7c4dff",
    "ROX1101": "#00e676",
    "KAW1101": "#ff6d00",
    "WHI2201": "#ff1744",
}

MODEL_KEYS = [
    "linear_regression",
    "catboost_regressor",
    "chronos2s_zeroshot",
    "chronos2s_finetune",
    "chronos2_full_zeroshot",
]

MODEL_LABELS = {
    "linear_regression": "Linear Regression",
    "catboost_regressor": "CatBoost",
    "chronos2s_zeroshot": "Chronos-2 Small Zero-Shot",
    "chronos2s_finetune": "Chronos-2 Small Fine-Tuned",
    "chronos2_full_zeroshot": "Chronos-2 Full Zero-Shot",
}

MODEL_COLORS = {
    "linear_regression": "#ff9100",
    "catboost_regressor": "#f0b429",
    "chronos2s_zeroshot": "#00d4ff",
    "chronos2s_finetune": "#7c4dff",
    "chronos2_full_zeroshot": "#00e676",
}

COLORS = {
    "bg": "#0a1628",
    "bg_secondary": "#0d1f3c",
    "primary": "#f0b429",
    "secondary": "#00d4ff",
    "text": "#e8e8e8",
    "text_muted": "#8899aa",
    "danger": "#ff4444",
    "success": "#00e676",
    "grid": "#1a2a4a",
}

SPIKE_COLORSCALE = [
    [0.0, "#0a1628"],
    [0.25, "#1a3a5c"],
    [0.5, "#f0b429"],
    [0.75, "#ff6d00"],
    [1.0, "#ff4444"],
]

CUSTOM_CSS = """
<style>
    div[data-testid="stMetric"] {
        background: rgba(13, 31, 60, 0.6);
        border: 1px solid rgba(240, 180, 41, 0.2);
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f0b429;
    }
</style>
"""

# --- Plotly template ---
VOLTAGE_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], size=12),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=20, t=40, b=40),
    )
)
pio.templates["voltage"] = VOLTAGE_TEMPLATE
pio.templates.default = "voltage"


# --- Data loading ---

@st.cache_data
def load_predictions(model: str, poc: str) -> pd.DataFrame:
    path = PREDICTIONS_DIR / f"{model}_{poc}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df.rename(columns={"node": "poc"})
    if hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dtype.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    for col in ("p10", "p90"):
        if col not in df.columns:
            df[col] = np.nan
    df["poc"] = poc
    return df[["timestamp", "poc", "horizon_h", "actual", "predicted", "p10", "p90"]]


@st.cache_data
def load_gold_panel() -> pd.DataFrame:
    df = pd.read_parquet(GOLD_PANEL_PATH)
    if hasattr(df["ts_utc"].dtype, "tz") and df["ts_utc"].dtype.tz is not None:
        df["ts_utc"] = df["ts_utc"].dt.tz_localize(None)
    df["POC"] = df["POC"].replace(GOLD_POC_MAP)
    return df


@st.cache_data
def compute_metrics(model: str, poc: str) -> dict:
    df = load_predictions(model, poc)
    if df.empty:
        return {"rmse": np.nan, "smape": np.nan}
    valid = df.dropna(subset=["actual", "predicted"])
    if valid.empty:
        return {"rmse": np.nan, "smape": np.nan}
    rmse = np.sqrt(((valid["actual"] - valid["predicted"]) ** 2).mean())
    denom = (valid["actual"].abs() + valid["predicted"].abs()) / 2
    denom = denom.clip(lower=1.0)
    smape = (np.abs(valid["actual"] - valid["predicted"]) / denom).mean()
    return {"rmse": round(float(rmse), 2), "smape": round(float(smape) * 100, 1)}


@st.cache_data
def build_leaderboard() -> pd.DataFrame:
    rows = []
    for model in MODEL_KEYS:
        for poc in POCS:
            m = compute_metrics(model, poc)
            rows.append({
                "Model": MODEL_LABELS[model],
                "Node": poc,
                "RMSE ($/MWh)": m["rmse"],
                "sMAPE (%)": m["smape"],
            })
    return pd.DataFrame(rows)


@st.cache_data
def get_spike_thresholds() -> dict:
    panel = load_gold_panel()
    train = panel[panel["ts_utc"] < "2025-01-01"]
    return {
        poc: float(train[train["POC"] == poc]["price_dmwh"].quantile(0.95))
        for poc in POCS
    }


@st.cache_data
def load_spike_predictions(poc: str) -> pd.DataFrame:
    path = PREDICTIONS_DIR / f"lr_smote_spike_{poc}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dtype.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        df["actual_spike"] = df["actual_spike"].astype(float)
        return df[["timestamp", "spike_prob", "actual_spike"]]
    return _generate_synthetic_spike_probs(poc)


@st.cache_data
def _generate_synthetic_spike_probs(poc: str) -> pd.DataFrame:
    panel = load_gold_panel()
    threshold = get_spike_thresholds()[poc]
    node = panel[panel["POC"] == poc].sort_values("ts_utc").reset_index(drop=True)
    test = node[node["ts_utc"] >= "2025-01-01"].copy()

    prices = test["price_dmwh"].values
    rev_max = pd.Series(prices[::-1]).rolling(48, min_periods=1).max().values
    fwd_max = np.roll(rev_max[::-1], -1)
    fwd_max[-1] = np.nan

    x = (fwd_max - threshold) / (threshold * 0.3)
    test["spike_prob"] = 1 / (1 + np.exp(-x))
    test["actual_spike"] = (fwd_max > threshold).astype(float)
    test.loc[test.index[-1], ["spike_prob", "actual_spike"]] = np.nan

    return (test[["ts_utc", "spike_prob", "actual_spike"]]
            .dropna()
            .rename(columns={"ts_utc": "timestamp"}))
