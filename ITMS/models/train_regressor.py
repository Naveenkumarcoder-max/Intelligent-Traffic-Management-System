# models/train_regressor.py

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import mlflow
import mlflow.sklearn

def train_signal_regressor():
    df = pd.read_csv('data/raw/traffic_data.csv')
    
    le_weather = joblib.load('models/le_weather.pkl')
    le_day = joblib.load('models/le_day.pkl')
    
    df['weather_enc'] = le_weather.transform(df['weather'])
    df['day_enc'] = le_day.transform(df['day_type'])
    
    features = ['hour', 'vehicle_count', 'avg_speed', 'weather_enc', 'day_enc']
    X = df[features]
    y = df['optimal_signal_time']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    mlflow.set_experiment("ITMS_Signal_Regressor")
    
    with mlflow.start_run():
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(model, "regressor_model")
        
        print(f"MAE: {mae:.4f} | R2 Score: {r2:.4f}")
    
    joblib.dump(model, 'models/signal_regressor.pkl')
    print("Regressor saved!")

train_signal_regressor()