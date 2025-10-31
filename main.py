from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib

# -------------------------------
# 🌤️ CONFIG
# -------------------------------
MODEL_PATH = "models/random_forest_weather.pkl"
Y_COLS = ['tempC', 'humidity', 'precipMM', 'cloudcover']

# -------------------------------
# ⚙️ Load Model
# -------------------------------


@st.cache_resource
def load_model(path):
    return joblib.load(path)


model = load_model(MODEL_PATH)

# -------------------------------
# 📄 PAGE SETUP
# -------------------------------
st.set_page_config(layout="wide")
st.title("📈 Tomorrow’s Weather Analysis")
st.markdown(
    "##### Comparing Model Predictions with OpenWeather API Forecasts for Tomorrow")
st.divider()

# -------------------------------
# ✅ USE SESSION DATA
# -------------------------------
if "today_data" not in st.session_state or st.session_state.today_data is None:
    st.warning(
        "⚠️ Please go back to the main page and fetch today's weather first.")
elif "tomorrow_data" not in st.session_state or st.session_state.tomorrow_data is None:
    st.warning("⚠️ Please predict tomorrow's weather from Page 2 first.")
else:
    df_today = st.session_state.today_data
    df_api = st.session_state.tomorrow_data

    st.subheader("📊 Model Performance vs API Forecasts")

    # Prepare features
    X_tomorrow = df_api.drop(['date_time'] + Y_COLS, axis=1)
    y_true = df_api[Y_COLS]

    # ✅ Ensure feature order matches training
    X_tomorrow = X_tomorrow[model.feature_names_in_]

    # Predict
    preds_tomorrow = model.predict(X_tomorrow)
    preds_tomorrow_df = pd.DataFrame(preds_tomorrow, columns=Y_COLS)
    preds_tomorrow_df['date_time'] = df_api['date_time']

    # Display metrics for tomorrow’s avg prediction
    avg_temp = preds_tomorrow_df["tempC"].mean()
    avg_humidity = preds_tomorrow_df["humidity"].mean()

    col1, col2 = st.columns(2)
    col1.metric("Predicted Avg Temperature (°C)", f"{avg_temp:.1f}")
    col2.metric("Predicted Avg Humidity (%)", f"{avg_humidity:.1f}")

    # Compact 2x2 grid layout for charts
    cols = st.columns(2)

    # Only show TempC and Humidity
    for i, col_name in enumerate(["tempC", "humidity"]):
        rmse = np.sqrt(
            ((y_true[col_name] - preds_tomorrow_df[col_name]) ** 2).mean())
        r2 = 1 - (((y_true[col_name] - preds_tomorrow_df[col_name]) ** 2).sum() /
                  ((y_true[col_name] - y_true[col_name].mean()) ** 2).sum())

        fig, ax = plt.subplots(figsize=(3, 2.2), dpi=100)
        ax.plot(df_api["date_time"], y_true[col_name],
                label="API Forecast", marker="o", linewidth=1)
        ax.plot(df_api["date_time"], preds_tomorrow_df[col_name],
                label="Model Prediction", linestyle="--", marker="x", linewidth=1)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        ax.set_title(f"{col_name.upper()} — Predicted vs API",
                     fontsize=8, pad=6)
        ax.set_xlabel("Time", fontsize=7)
        ax.set_ylabel(col_name, fontsize=7)
        ax.tick_params(axis='both', labelsize=6)
        ax.legend(fontsize=6, loc="best")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        plt.tight_layout(pad=1.0)

        with cols[i % 2]:
            st.markdown(
                f"**{col_name.upper()}** — RMSE: `{rmse:.2f}` | R²: `{r2:.3f}`",
                unsafe_allow_html=True,
            )
            st.pyplot(fig, use_container_width=False)

# -------------------------------
# 📌 FIXED FOOTER
# -------------------------------
st.markdown("""
<style>
footer {
    visibility: hidden;
}
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0e1117;
    color: white;
    text-align: center;
    padding: 10px 0;
    font-size: 0.9em;
}
</style>

<div class="footer">
    © 2025 <b>Aaryan Dawalkar</b> — Weather Prediction App | powered by <b>Streamlit</b>
</div>
""", unsafe_allow_html=True)
