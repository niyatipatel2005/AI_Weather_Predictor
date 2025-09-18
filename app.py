# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# ==============================
# 1. Load Trained Model
# ==============================
MODEL_PATH = "weather_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Trained model not found! Please run model.py first.")
    st.stop()

data = joblib.load(MODEL_PATH)
model = data["model"]
feature_names = data["features"]

# ==============================
# 2. App Config
# ==============================
st.set_page_config(page_title="AI Weather Predictor", page_icon="🌤️", layout="centered")

st.title("🌤️ AI Weather Predictor")
st.markdown("Use this app to predict **Temperature (°C)** based on weather conditions.")
st.divider()

# ==============================
# 3. Sidebar
# ==============================
st.sidebar.title("⚙️ Settings")
st.sidebar.info("This app uses **manual input** to predict temperature.")

# ==============================
# 4. Manual Input
# ==============================
st.header("📥 Enter Weather Conditions")

input_data = pd.DataFrame([[0]*len(feature_names)], columns=feature_names)

apparent_temp = st.number_input("Apparent Temperature (°C)", min_value=-50.0, max_value=50.0, value=0.0, step=0.1)
humidity = st.slider("Humidity", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=0.0, step=0.1)
wind_bearing = st.number_input("Wind Bearing (degrees)", min_value=0.0, max_value=360.0, value=0.0, step=1.0)
visibility = st.number_input("Visibility (km)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
pressure = st.number_input("Pressure (millibars)", min_value=900.0, max_value=1100.0, value=1013.0, step=0.1)

# Fill input dataframe safely
if "Apparent Temperature (C)" in input_data.columns:
    input_data["Apparent Temperature (C)"] = apparent_temp
if "Humidity" in input_data.columns:
    input_data["Humidity"] = humidity
if "Wind Speed (km/h)" in input_data.columns:
    input_data["Wind Speed (km/h)"] = wind_speed
if "Wind Bearing (degrees)" in input_data.columns:
    input_data["Wind Bearing (degrees)"] = wind_bearing
if "Visibility (km)" in input_data.columns:
    input_data["Visibility (km)"] = visibility
if "Pressure (millibars)" in input_data.columns:
    input_data["Pressure (millibars)"] = pressure

# ==============================
# 5. Predict Button
# ==============================
if st.button("🌡️ Predict Temperature"):
    prediction = model.predict(input_data)[0]

    st.divider()
    st.header("🔮 Prediction Result")
    st.success("✅ Prediction Complete!")
    st.markdown(
        f"""
        <div style="padding:20px; border-radius:10px; background-color:#f0f8ff; text-align:center">
            <h2 style="color:#ff6600;">🌡️ Predicted Temperature: <b>{prediction:.2f} °C</b></h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("📋 Input Summary"):
        st.write(input_data)
