# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("models/xgboost_weather.pkl")

st.title("🌤️ Weather Prediction App")
st.write("Predict tomorrow's temperature, humidity, rainfall, and cloud cover!")

# Input fields (you can add more based on your dataset)
sunrise = st.number_input(
    "Sunrise (in hours, e.g., 7.25 for 7:15 AM)", value=7.25)
sunset = st.number_input(
    "Sunset (in hours, e.g., 17.5 for 5:30 PM)", value=17.5)
humidity = st.number_input("Today's Humidity (%)", value=60)
pressure = st.number_input("Pressure (hPa)", value=1013)
windspeed = st.number_input("Wind Speed (km/h)", value=10)
precip = st.number_input("Precipitation (mm)", value=0.0)

# Predict button
if st.button("Predict Tomorrow's Weather"):
    # Prepare input dataframe
    input_data = pd.DataFrame([{
        "sunrise": sunrise,
        "sunset": sunset,
        "humidity": humidity,
        "pressure": pressure,
        "windspeedKmph": windspeed,
        "precipMM": precip
    }])

    # Predict using your model
    preds = model.predict(input_data)

    st.subheader("🌈 Predicted Values for Tomorrow:")
    st.write(f"**Temperature (°C):** {preds[0][0]:.2f}")
    st.write(f"**Humidity (%):** {preds[0][1]:.2f}")
    st.write(f"**Rainfall (mm):** {preds[0][2]:.2f}")
    st.write(f"**Cloud Cover (%):** {preds[0][3]:.2f}")
