import pandas as pd
import numpy as np

np.random.seed(42)
n = 5000

def generate_traffic_data():
    hours = np.random.randint(0, 24, n)
    vehicle_count = np.random.randint(10, 500, n)
    avg_speed = np.random.uniform(5, 80, n)
    weather = np.random.choice(['clear', 'rain', 'fog'], n)
    day_type = np.random.choice(['weekday', 'weekend'], n)
    
    # Congestion label logic
    congestion = []
    for i in range(n):
        if vehicle_count[i] > 350 or avg_speed[i] < 20:
            congestion.append('high')
        elif vehicle_count[i] > 200 or avg_speed[i] < 40:
            congestion.append('medium')
        else:
            congestion.append('low')
    
    # Signal timing logic (regression target)
    signal_time = []
    for i in range(n):
        base = 30
        if congestion[i] == 'high':
            base = 90
        elif congestion[i] == 'medium':
            base = 60
        noise = np.random.randint(-5, 5)
        signal_time.append(base + noise)
    
    df = pd.DataFrame({
        'hour': hours,
        'vehicle_count': vehicle_count,
        'avg_speed': avg_speed,
        'weather': weather,
        'day_type': day_type,
        'congestion_level': congestion,
        'optimal_signal_time': signal_time
    })
    
    df.to_csv('data/raw/traffic_data.csv', index=False)
    print("Dataset generated successfully!")
    return df

generate_traffic_data()