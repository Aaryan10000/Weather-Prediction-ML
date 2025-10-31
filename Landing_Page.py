import streamlit as st

st.set_page_config(page_title="Weather Prediction App", layout="wide")

st.title("🌦️ Real-time Weather Predictor")
st.markdown("#### Powered by XGBoost + OpenWeatherMap API")
st.markdown("---")

st.markdown("""
Welcome to the **Weather Prediction App** 🌤️  
This app predicts **tomorrow’s temperature and humidity** using real-time weather data and an ML model.
""")

st.image("assets/Image.png", width=200)

if st.button("🚀 Get Started"):
    st.switch_page("pages/1_Current_Weather.py")

# --- Footer ---
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
