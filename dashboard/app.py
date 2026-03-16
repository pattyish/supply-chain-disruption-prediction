import streamlit as st
import requests

st.set_page_config(page_title="Supply Chain Disruption Monitor")
st.title("Supply Chain Disruption Monitor")

shipping_pressure = st.slider("Shipping Pressure", 0.0, 5.0, 1.0)
port_wait = st.number_input("Port wait time (hours)", min_value=0.0, max_value=240.0, value=2.0)
weather_risk = st.selectbox("Weather risk", [0, 1, 2])
distance = st.number_input("Distance (km)", min_value=0.0, value=1500.0)

if st.button("Predict Risk"):
    try:
        resp = requests.post("http://localhost:8000/predict", json={
            "shipping_pressure": shipping_pressure,
            "port_wait_time": port_wait,
            "weather_risk": weather_risk,
            "distance": distance
        }, timeout=5.0)
        st.write(resp.json())
    except Exception as e:
        st.error("Prediction request failed: {}".format(e))
