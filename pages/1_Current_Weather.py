from datetime import datetime
import streamlit as st
import pandas as pd
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
    st.session_state.today_data = None


def get_today_weather(city=CITY):
    URL = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(URL)
    if response.status_code != 200:
        st.error("❌ Failed to fetch current weather data.")
        return None
    data = response.json()
    sunrise = datetime.fromtimestamp(
        data["sys"]["sunrise"]).hour + datetime.fromtimestamp(data["sys"]["sunrise"]).minute / 60
    sunset = datetime.fromtimestamp(
        data["sys"]["sunset"]).hour + datetime.fromtimestamp(data["sys"]["sunset"]).minute / 60
    now = datetime.now()

    today_data = {
        "maxtempC": data["main"]["temp_max"],
        "mintempC": data["main"]["temp_min"],
        "sunHour": 8.0,
        "uvIndex": 5,
        "sunrise": round(sunrise, 2),
        "sunset": round(sunset, 2),
        "DewPointC": data["main"]["temp"] - ((100 - data["main"]["humidity"]) / 5),
        "WindGustKmph": data["wind"]["speed"] * 3.6,
        "pressure": data["main"]["pressure"],
        "visibility": data.get("visibility", 10000) / 1000,
        "winddirDegree": data["wind"].get("deg", 0),
        "windspeedKmph": data["wind"]["speed"] * 3.6,
        "year": now.year, "month": now.month, "day": now.day,
        "hour": now.hour, "dayofweek": now.weekday(),
    }
    return pd.DataFrame([today_data])


st.title("🌤️ Today's Weather")

city = st.text_input("Enter City Name:", "New Delhi")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔍 Show Today's Weather"):
        st.session_state.today_data = get_today_weather(city)
        df_today = st.session_state.today_data
        if df_today is not None:
            n_cols = len(df_today.columns)
            split_size = n_cols // 3
            df_part1 = df_today.iloc[:, :split_size]
            df_part2 = df_today.iloc[:, split_size:2*split_size]
            df_part3 = df_today.iloc[:, 2*split_size:]

            st.subheader("📅 Today's Weather Data:")
            st.dataframe(df_part1)
            st.dataframe(df_part2)
            st.dataframe(df_part3)
with col2:
    if st.button("🔄 Refresh Data"):
        st.session_state.today_data = get_today_weather(city)
        st.success("✅ Weather data refreshed!")

if st.session_state.today_data is not None:
    if st.button("🤖 Predict Tomorrow's Weather"):
        st.switch_page("pages/2_Tomorrow_Predicted_Weather.py")

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
