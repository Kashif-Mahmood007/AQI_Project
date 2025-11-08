
# ==========================
# 🌤️ AQI Forecast Streamlit App — Local Models + Hopsworks Features
# ==========================

import os
import json
import streamlit as st
import hopsworks
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------
# ⚙️ Page Config
# ------------------------------------------
st.set_page_config(page_title="AQI Forecast Dashboard", layout="wide")

st.title("🌤️ Air Quality Forecast Dashboard")
st.markdown("""
This app connects to **Hopsworks** to fetch the latest features and uses **locally stored trained models**
to predict the next 24-hour AQI values.
""")

# =========================================================
# 🧭 Sidebar — Login Section
# =========================================================
st.sidebar.header("🔑 Hopsworks Login")

api_key = st.sidebar.text_input("API Key", type="password")
project_name = st.sidebar.text_input("Project Name", value="AQI_Project_10Pearls")
fg_name = "aqi_scaled_features_10pearls"
fg_version = 1

# =========================================================
# Helper Functions
# =========================================================
def aqi_category(aqi):
    aqi = float(aqi)
    if aqi <= 50:
        return "Good", "#55A84F"
    if aqi <= 100:
        return "Moderate", "#A3C853"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#FFF833"
    if aqi <= 200:
        return "Unhealthy", "#F29C33"
    if aqi <= 300:
        return "Very Unhealthy", "#E93F33"
    return "Hazardous", "#AF2D24"


# =========================================================
# 🔐 Login to Hopsworks
# =========================================================
if st.sidebar.button("Connect"):
    try:
        project = hopsworks.login(api_key_value=api_key, project=project_name)
        fs = project.get_feature_store()
        st.session_state["fs"] = fs
        st.success("✅ Connected to Hopsworks!")
    except Exception as e:
        st.error(f"❌ Login failed: {e}")

# =========================================================
# 🚀 Main Logic — only after successful login
# =========================================================
if "fs" in st.session_state:

    fs = st.session_state["fs"]

    # ------------------------------------------
    # 📦 Load Models and Preprocessing Pipeline
    # ------------------------------------------
    st.subheader("📦 Loading Local Models")

    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

    try:
        base_model_path = os.path.join(models_dir, "best_model.pkl")
        forecast_model_path = os.path.join(models_dir, "best_forecast_model.pkl")

        with open(base_model_path, "rb") as f:
            base_model = pickle.load(f)
        with open(forecast_model_path, "rb") as f:
            forecast_model = pickle.load(f)

        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Could not load models or pipeline: {e}")
        st.stop()

    # ------------------------------------------
    # 🧩 Load feature_order.json
    # ------------------------------------------
    feature_order_path = os.path.join(models_dir, "feature_order.json")
    if os.path.exists(feature_order_path):
        with open(feature_order_path, "r") as f:
            feature_order = json.load(f)
        # st.info(f"✅ Loaded feature_order.json ({len(feature_order)} features)")
    else:
        st.error("❌ feature_order.json not found. Please place it under 'models/' folder.")
        st.stop()

    # ------------------------------------------
    # 📊 Fetch Latest Features from Hopsworks
    # ------------------------------------------
    st.subheader("📊 Fetching Latest Feature Data")

    try:
        fg = fs.get_feature_group(fg_name, version=fg_version)
        df_scaled = fg.read()
        if "timestamp" in df_scaled.columns:
            df_scaled["timestamp"] = pd.to_datetime(df_scaled["timestamp"])
        df_scaled = df_scaled.sort_values("timestamp").reset_index(drop=True)

        st.success(f"✅ Loaded {len(df_scaled)} rows of feature data.")
    except Exception as e:
        st.error(f"⚠️ Could not fetch feature data: {e}")
        st.stop()

    # ------------------------------------------
    # 🔮 24-hour Forecast Prediction
    # ------------------------------------------

    st.subheader("🌟 AQI Forecast (Next 24 Hours)")

    last_row = df_scaled.iloc[-1]
    FEATURES = feature_order
    HORIZON = 72

    try:
        # Ensure all required features exist (fill missing ones)
        for col in FEATURES:
            if col not in last_row.index:
                st.warning(f"⚠️ Missing feature '{col}' in fetched data — filling with 0.0")
                last_row[col] = 0.0

        # Keep only the required features in correct order
        X_last = pd.DataFrame([last_row[FEATURES].values], columns=FEATURES)
        X_last = X_last.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Predict using trained forecast model
        preds_72 = forecast_model.predict(X_last).flatten()
        

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        st.stop()


    # ------------------------------------------
    # 🕒 Create future timestamps
    # ------------------------------------------
    base_ts = last_row["timestamp"]
    future_timestamps = [base_ts + timedelta(hours=i) for i in range(1, HORIZON + 1)]

    forecast_df = pd.DataFrame({"timestamp": future_timestamps, "predicted_aqi": preds_72})
    forecast_df["category"], forecast_df["color"] = zip(*[aqi_category(x) for x in forecast_df["predicted_aqi"]])

    # ------------------------------------------
    # 📊 Dashboard Metrics
    # ------------------------------------------
    current_aqi = float(last_row["aqi"]) if "aqi" in df_scaled.columns else np.nan
    next_hour = float(preds_72[0])
    max_24 = float(np.max(preds_72))
    avg_24 = float(np.mean(preds_72))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current AQI", f"{current_aqi:.1f}")
    m2.metric("Next-hour forecast", f"{next_hour:.1f}")
    m3.metric("24h Max (pred)", f"{max_24:.1f}")
    m4.metric("24h Avg (pred)", f"{avg_24:.1f}")

    # ------------------------------------------
    # 📈 Visualization
    # ------------------------------------------
    st.subheader("📈 Recent AQI & 24-Hour Forecast")

    hist_df = df_scaled[["timestamp", "aqi"]].dropna().tail(72).rename(columns={"aqi": "value"})
    hist_df["series"] = "Observed"
    hist_df["category"], hist_df["color"] = zip(*[aqi_category(x) for x in hist_df["value"]])

    fc_df = forecast_df.rename(columns={"predicted_aqi": "value"})
    fc_df["series"] = "Forecast"

    plot_df = pd.concat([hist_df, fc_df], ignore_index=True)

    fig = px.line(plot_df, x="timestamp", y="value", color="series", markers=True,
                  hover_data=["category"], title="AQI — Observed vs Forecasted")

    forecast_markers = go.Scatter(
        x=fc_df["timestamp"], y=fc_df["value"], mode="markers+lines", name="Forecast",
        marker=dict(color=fc_df["color"], size=8, line=dict(width=1))
    )
    fig.add_trace(forecast_markers)
    fig.update_layout(xaxis_title="Timestamp", yaxis_title="AQI", hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True)


    # ------------------------------------------
    # 📋 Forecasted AQI Values + Download
    # ------------------------------------------
    st.subheader("📋 Forecasted AQI Values")
    display_df = forecast_df[["timestamp", "predicted_aqi", "category"]].copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(display_df.reset_index(drop=True))

    csv = forecast_df.to_csv(index=False)
    st.download_button("Download Forecast CSV", csv, file_name="aqi_forecast_24h.csv")
