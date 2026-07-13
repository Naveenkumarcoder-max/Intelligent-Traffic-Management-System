"""
ITMS - Inference / Prediction Script

Loads the best trained model + scaler from `models/` and predicts congestion
for new traffic data. Use this inside your FastAPI endpoint or Streamlit app.

Usage:
    python predict.py --data data/new_readings.csv --models-dir models
"""

import argparse
import os
import joblib
import pandas as pd

from train_models import engineer_features  # reuse the same feature pipeline


def load_best_model(models_dir: str):
    with open(os.path.join(models_dir, "best_model.txt")) as f:
        best_name = f.read().strip()
    model_path = os.path.join(models_dir, f"{best_name}.pkl")
    model = joblib.load(model_path)
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    print(f"Loaded best model: {best_name}")
    return model, scaler


def predict(df: pd.DataFrame, model, scaler) -> pd.Series:
    df_feat = engineer_features(df)
    # Align columns with what the scaler was fit on
    df_feat = df_feat.reindex(columns=scaler.feature_names_in_, fill_value=0)
    X_scaled = scaler.transform(df_feat)
    return pd.Series(model.predict(X_scaled), index=df.index, name="predicted_congestion")


def main():
    parser = argparse.ArgumentParser(description="Run ITMS congestion predictions")
    parser.add_argument("--data", required=True, help="Path to CSV with new traffic readings")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    model, scaler = load_best_model(args.models_dir)
    preds = predict(df, model, scaler)

    out = df.copy()
    out["predicted_congestion"] = preds
    out.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")
    print(out.head())


if __name__ == "__main__":
    main()
