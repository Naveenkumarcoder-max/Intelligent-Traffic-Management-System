# api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ITMS Prediction API")

# Load models
classifier = joblib.load('models/congestion_classifier.pkl')
regressor = joblib.load('models/signal_regressor.pkl')
le_weather = joblib.load('models/le_weather.pkl')
le_day = joblib.load('models/le_day.pkl')
le_congestion = joblib.load('models/le_congestion.pkl')

class TrafficInput(BaseModel):
    hour: int
    vehicle_count: int
    avg_speed: float
    weather: str        # 'clear', 'rain', 'fog'
    day_type: str       # 'weekday', 'weekend'

@app.get("/")
def home():
    return {"message": "ITMS API is running!"}

@app.post("/predict")
def predict(data: TrafficInput):
    weather_enc = le_weather.transform([data.weather])[0]
    day_enc = le_day.transform([data.day_type])[0]
    
    features = [[
        data.hour,
        data.vehicle_count,
        data.avg_speed,
        weather_enc,
        day_enc
    ]]
    
    congestion_enc = classifier.predict(features)[0]
    congestion_label = le_congestion.inverse_transform([congestion_enc])[0]
    signal_time = regressor.predict(features)[0]
    
    return {
        "congestion_level": congestion_label,
        "recommended_signal_time_seconds": round(signal_time, 2),
        "status": "success"
    }