import streamlit as st
st.set_page_config(page_title="NZ Power Pulse — Predict", page_icon="⚡", layout="wide")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import app_config as cfg

st.markdown(cfg.CUSTOM_CSS, unsafe_allow_html=True)

st.title("Interactive Price Prediction")

MODELS_DIR = Path(__file__).parent.parent / "data" / "models"

@st.cache_resource
def load_models():
    lr = joblib.load(MODELS_DIR / "linear_regression.joblib")
    lr_scaler = joblib.load(MODELS_DIR / "linear_scaler.joblib")
    cb = joblib.load(MODELS_DIR / "catboost_regressor.joblib")
    spike_clf = joblib.load(MODELS_DIR / "lr_smote_classifier.joblib")
    return lr, lr_scaler, cb, spike_clf

lr_model, lr_scaler, cb_model, spike_clf = load_models()

# --- Shared inputs in sidebar ---
st.sidebar.header("Scenario Inputs")

dt = st.sidebar.date_input("Date", value=pd.Timestamp("2025-06-15"))
hour = st.sidebar.slider("Hour", 0, 23, 12)
minute = st.sidebar.selectbox("Minute", [0, 30], index=0)

ts = pd.Timestamp(dt) + pd.Timedelta(hours=hour, minutes=minute)
day_of_week = ts.dayofweek
month = ts.month
is_weekend = int(day_of_week >= 5)
hour_val = hour + minute / 60.0
hour_sin = np.sin(2 * np.pi * hour_val / 24)
hour_cos = np.cos(2 * np.pi * hour_val / 24)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
dow_sin = np.sin(2 * np.pi * day_of_week / 7)
dow_cos = np.cos(2 * np.pi * day_of_week / 7)

st.sidebar.subheader("Weather")
temp_c = st.sidebar.slider("Temperature (°C)", -10.0, 35.0, 13.0, 0.5)
wind_ms = st.sidebar.slider("Wind speed (m/s)", 0.0, 20.0, 3.0, 0.5)
solar_wm2 = st.sidebar.slider("Solar radiation (W/m²)", 0.0, 1100.0, 100.0, 10.0)
rain_mm = st.sidebar.slider("Rainfall (mm/30min)", 0.0, 25.0, 0.0, 0.5)

heating_degrees = max(0.0, 15.0 - temp_c)
cooling_degrees = max(0.0, temp_c - 22.0)

st.sidebar.subheader("Price Lags")
price_lag_1h = st.sidebar.number_input("Price 1h ago ($/MWh)", -100.0, 6000.0, 138.0, 10.0)
price_lag_24h = st.sidebar.number_input("Price 24h ago ($/MWh)", -100.0, 6000.0, 138.0, 10.0)
price_lag_168h = st.sidebar.number_input("Price 1 week ago ($/MWh)", -100.0, 6000.0, 138.0, 10.0)

# --- Spike Risk ---
spike_features = pd.DataFrame([{
    "hour_sin": hour_sin, "hour_cos": hour_cos,
    "month_sin": month_sin, "month_cos": month_cos,
    "dow_sin": dow_sin, "dow_cos": dow_cos,
    "is_weekend": is_weekend,
    "temp_c": temp_c, "wind_ms": wind_ms,
    "solar_wm2": solar_wm2, "rain_mm": rain_mm,
    "heating_degrees": heating_degrees, "cooling_degrees": cooling_degrees,
    "price_lag_1h": price_lag_1h, "price_lag_24h": price_lag_24h,
    "price_lag_168h": price_lag_168h,
}])

spike_prob = spike_clf.predict_proba(spike_features)[0, 1]

if spike_prob >= 0.7:
    spike_color, spike_label = cfg.COLORS["danger"], "HIGH"
elif spike_prob >= 0.3:
    spike_color, spike_label = "#ff9100", "MODERATE"
else:
    spike_color, spike_label = cfg.COLORS["success"], "LOW"

st.markdown(
    f"**Spike Risk:** "
    f"<span style='color:{spike_color};font-weight:bold'>{spike_label}</span> "
    f"— {spike_prob:.1%} probability price exceeds $300/MWh",
    unsafe_allow_html=True,
)

st.divider()

# --- Tabs ---
tab_lr, tab_cb = st.tabs(["Linear Regression", "CatBoost"])

with tab_lr:
    st.subheader("Linear Regression")
    demand = st.slider("Demand (MWh)", 0.0, 80.0, 14.0, 0.5, key="lr_demand")

    lr_features = pd.DataFrame([{
        "DemandMWh": demand,
        "temp_c": temp_c,
        "solar_wm2": solar_wm2,
        "rain_mm": rain_mm,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "is_weekend": is_weekend,
        "price_lag_1h": price_lag_1h,
        "price_lag_24h": price_lag_24h,
        "price_lag_168h": price_lag_168h,
    }])

    lr_scaled = lr_scaler.transform(lr_features)
    lr_pred = lr_model.predict(lr_scaled)[0]

    st.metric("Predicted Price", f"${lr_pred:.2f}/MWh")

    with st.expander("Feature values"):
        st.dataframe(lr_features.T.rename(columns={0: "Value"}).astype(str), width=400)

with tab_cb:
    st.subheader("CatBoost")
    node = st.selectbox(
        "Grid node", cfg.POCS,
        format_func=lambda p: f"{p} — {cfg.POC_LABELS[p]}",
        key="cb_node",
    )

    cb_features = pd.DataFrame([{
        "node": node,
        "temp_c": temp_c,
        "wind_ms": wind_ms,
        "solar_wm2": solar_wm2,
        "rain_mm": rain_mm,
        "hour": hour,
        "minute": minute,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "price_lag_1h": price_lag_1h,
        "price_lag_24h": price_lag_24h,
        "price_lag_168h": price_lag_168h,
        "heating_degrees": heating_degrees,
        "cooling_degrees": cooling_degrees,
    }])

    cb_pred = cb_model.predict(cb_features)[0]

    st.metric("Predicted Price", f"${cb_pred:.2f}/MWh")

    with st.expander("Feature values"):
        st.dataframe(cb_features.T.rename(columns={0: "Value"}).astype(str), width=400)
