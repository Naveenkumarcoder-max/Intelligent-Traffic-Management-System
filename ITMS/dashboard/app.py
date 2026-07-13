# dashboard/app.py

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import random


st.set_page_config(page_title="ITMS Dashboard", layout="wide")
st.title("🚦 Intelligent Traffic Management System")

# Sidebar inputs
st.sidebar.header("Traffic Input Parameters")
hour = st.sidebar.slider("Hour of Day", 0, 23, 8)
vehicle_count = st.sidebar.slider("Vehicle Count", 10, 500, 200)
avg_speed = st.sidebar.slider("Average Speed (km/h)", 5.0, 80.0, 40.0)
weather = st.sidebar.selectbox("Weather", ['clear', 'rain', 'fog'])
day_type = st.sidebar.selectbox("Day Type", ['weekday', 'weekend'])

if st.sidebar.button("Predict Now"):
    payload = {
        "hour": hour,
        "vehicle_count": vehicle_count,
        "avg_speed": avg_speed,
        "weather": weather,
        "day_type": day_type
    }
    
    response = requests.post("http://localhost:8000/predict", json=payload)
    result = response.json()
    
    col1, col2 = st.columns(2)
    
    with col1:
        congestion = result['congestion_level']
        color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(congestion, "⚪")
        st.metric(label="Congestion Level", value=f"{color} {congestion.upper()}")
    
    with col2:
        st.metric(
            label="Recommended Signal Time",
            value=f"{result['recommended_signal_time_seconds']} sec"
        )

# Simulated live traffic chart
st.subheader("📊 Live Traffic Volume (Simulated)")
hours_list = list(range(24))
volume = [random.randint(50, 450) for _ in hours_list]
fig = px.line(
    x=hours_list, y=volume,
    labels={'x': 'Hour', 'y': 'Vehicle Count'},
    title="Hourly Traffic Volume"
)
st.plotly_chart(fig, use_container_width=True)