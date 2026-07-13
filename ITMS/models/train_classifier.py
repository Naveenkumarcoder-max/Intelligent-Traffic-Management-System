# models/train_classifier.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import mlflow
import mlflow.sklearn

def train_congestion_classifier():
    # Load data
    df = pd.read_csv('data/raw/traffic_data.csv')
    
    # Encode categorical columns
    le_weather = LabelEncoder()
    le_day = LabelEncoder()
    le_congestion = LabelEncoder()
    
    df['weather_enc'] = le_weather.fit_transform(df['weather'])
    df['day_enc'] = le_day.fit_transform(df['day_type'])
    df['congestion_enc'] = le_congestion.fit_transform(df['congestion_level'])
    
    # Features and target
    features = ['hour', 'vehicle_count', 'avg_speed', 'weather_enc', 'day_enc']
    X = df[features]
    y = df['congestion_enc']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # MLflow tracking
    mlflow.set_experiment("ITMS_Congestion_Classifier")
    
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "classifier_model")
        
        print(f"Classifier Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
    
    # Save model and encoders
    joblib.dump(model, 'models/congestion_classifier.pkl')
    joblib.dump(le_weather, 'models/le_weather.pkl')
    joblib.dump(le_day, 'models/le_day.pkl')
    joblib.dump(le_congestion, 'models/le_congestion.pkl')
    
    print("Classifier saved!")

train_congestion_classifier()