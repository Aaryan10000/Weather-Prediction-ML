from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib


st.set_page_config(layout="wide")

API_KEY = "60c75cad07fdda2ae13c40d8d6dd3241"
CITY = "Delhi"
MODEL_PATH = "models/random_forest_weather.pkl"
Y_COLS = ['tempC', 'humidity', 'precipMM', 'cloudcover']


@st.cache_resource
def load_model(path):
    return joblib.load(path)


model = load_model(MODEL_PATH)

if "today_data" not in st.session_state:
    st.warning("⚠️ Please fetch today's weather first.")
    st.stop()


def prepare_features_from_api(df_api):
    """Rebuilds model input features from API forecast data."""
    df = df_api.copy()

    # Convert date_time to components
    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month
    df['day'] = df['date_time'].dt.day
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek

    # Add placeholder or computed values for features not in forecast API
    df['maxtempC'] = df['tempC']
    df['mintempC'] = df['tempC']
    df['sunHour'] = 8.0
    df['uvIndex'] = 5
    df['sunrise'] = 6.0
    df['sunset'] = 18.0
    df['DewPointC'] = df['tempC'] - ((100 - df['humidity']) / 5)
    df['WindGustKmph'] = 10.0
    df['pressure'] = 1010
    df['visibility'] = 10
    df['winddirDegree'] = 90
    df['windspeedKmph'] = 10

    # Drop unwanted
    drop_cols = ['tempC', 'humidity', 'precipMM', 'cloudcover', 'date_time']
    df = df.drop(
        columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    return df


def get_tomorrow_forecast(city=CITY):
    URL = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(URL)
    if response.status_code != 200:
        st.error("❌ Failed to fetch forecast data.")
        return None
    data = response.json()
    forecast_list = data.get("list", [])
    city_info = data["city"]
    sunrise = datetime.fromtimestamp(
        city_info["sunrise"]).hour + datetime.fromtimestamp(city_info["sunrise"]).minute / 60
    sunset = datetime.fromtimestamp(
        city_info["sunset"]).hour + datetime.fromtimestamp(city_info["sunset"]).minute / 60

    tomorrow = (datetime.utcnow() + timedelta(days=1)).date()
    forecasts = []
    for f in forecast_list:
        dt = datetime.fromtimestamp(f["dt"])
        if dt.date() == tomorrow:
            main = f["main"]
            wind = f.get("wind", {})
            forecasts.append({
                "date_time": dt,
                "tempC": main["temp"],
                "humidity": main["humidity"],
                "precipMM": f.get("rain", {}).get("3h", 0),
                "cloudcover": f["clouds"]["all"]
            })
    return pd.DataFrame(forecasts)


st.title("📊 Tomorrow's Weather Prediction")

df_today = st.session_state.today_data
preds = model.predict(df_today)
preds_df = pd.DataFrame(preds, columns=Y_COLS)

col1, col2 = st.columns(2)
col1.metric("Predicted Temperature (°C)", f"{preds_df['tempC'][0]:.1f}")
col2.metric("Predicted Humidity (%)", f"{preds_df['humidity'][0]:.1f}")

df_api = get_tomorrow_forecast(CITY)
if df_api is not None and not df_api.empty:

    X_tomorrow = prepare_features_from_api(df_api)
    X_tomorrow = X_tomorrow[model.feature_names_in_]
    preds_tomorrow = model.predict(X_tomorrow)
    y_true = df_api[Y_COLS]

    # preds_tomorrow = model.predict(df_api.drop(
    #     ['tempC', 'humidity', 'precipMM', 'cloudcover'], axis=1, errors='ignore'))

    preds_tomorrow_df = pd.DataFrame(preds_tomorrow, columns=Y_COLS)
    preds_tomorrow_df['date_time'] = df_api['date_time']

    st.subheader("📈 Model vs API Forecasts")


# Compact 2x2 grid layout for charts
cols = st.columns(2)

# Only show TempC and Humidity
for i, col_name in enumerate(["tempC", "humidity"]):
    if col_name not in y_true.columns or col_name not in preds_tomorrow_df.columns:
        st.warning(f"⚠️ Column '{col_name}' missing in data — skipping plot.")
        continue

    # Remove NaN for safe metric computation
    valid_idx = (~y_true[col_name].isna()) & (
        ~preds_tomorrow_df[col_name].isna())
    y_t = y_true.loc[valid_idx, col_name]
    y_p = preds_tomorrow_df.loc[valid_idx, col_name]

    if len(y_t) == 0:
        st.warning(f"⚠️ No valid data points for {col_name}.")
        continue

    rmse = np.sqrt(((y_t - y_p) ** 2).mean())
    r2 = 1 - (((y_t - y_p) ** 2).sum() /
              ((y_t - y_t.mean()) ** 2).sum())

    # Plot setup
    fig, ax = plt.subplots(figsize=(3, 2.3), dpi=120)
    ax.plot(df_api["date_time"], y_t,
            label="API Forecast", marker="o", linewidth=1)
    ax.plot(df_api["date_time"], y_p, label="Model Prediction",
            linestyle="--", marker="x", linewidth=1)

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_title(f"{col_name.upper()} — Predicted vs API", fontsize=9, pad=6)
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
