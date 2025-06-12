import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression

def load_and_preprocess_data(file_path="hourly_data_lahore_2020_2025.csv"):
    """Load and preprocess data with AQI-inspired features."""    
    data = pd.read_csv(file_path)
        
    data = data.sort_values(by='Datetime')
        
    data['Date'] = pd.to_datetime(data['Datetime'])
        
    data['Hour'] = data['Date'].dt.hour
    data['Day_of_Week'] = data['Date'].dt.weekday
    data['Month'] = data['Date'].dt.month
    data['Is_Peak_Hour'] = data['Hour'].apply(lambda x: 1 if 18 <= x <= 22 else 0)
    data['Is_Weekend'] = data['Day_of_Week'].apply(lambda x: 1 if x in [5, 6] else 0)
    
    data['Lag_1_Power'] = data['Global_active_power'].shift(1).fillna(data['Global_active_power'].mean())
    data['Lag_2_Power'] = data['Global_active_power'].shift(2).fillna(data['Global_active_power'].mean())
    data['Lag_3_Power'] = data['Global_active_power'].shift(3).fillna(data['Global_active_power'].mean())
    
    data['Power_Diff'] = data['Global_active_power'].diff().fillna(0)
    data['Voltage_Diff'] = data['Voltage'].diff().fillna(0)

    data['Num_Fridges'] = np.random.randint(1, 3, len(data))
    data['Num_Washing_Machines'] = np.random.randint(0, 2, len(data))
    data['Num_ACs'] = np.random.randint(0, 3, len(data))
    data['Num_Fans'] = np.random.randint(1, 5, len(data))
    data['Num_TVs'] = np.random.randint(1, 3, len(data))

    data['Power_Fridges'] = data['Num_Fridges'] * 0.2
    data['Power_Washing_Machines'] = data['Num_Washing_Machines'] * 0.5
    data['Power_ACs'] = data['Num_ACs'] * 1.5
    data['Power_Fans'] = data['Num_Fans'] * 0.075
    data['Power_TVs'] = data['Num_TVs'] * 0.1

    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
    data['Day_of_Week_sin'] = np.sin(2 * np.pi * data['Day_of_Week'] / 7)
    data['Day_of_Week_cos'] = np.cos(2 * np.pi * data['Day_of_Week'] / 7)

    data['Rolling_Mean_Power'] = data['Global_active_power'].rolling(window=3).mean().fillna(data['Global_active_power'].mean())
    data['Rolling_Mean_Power_6'] = data['Global_active_power'].rolling(window=6).mean().fillna(data['Global_active_power'].mean())

    data['Temp_Humidity_Ratio'] = data['Temperature'] / (data['Humidity'] + 1)  # Avoid division by zero

    features = [
        'Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
        'Hour', 'Day_of_Week', 'Is_Peak_Hour', 'Month', 'Is_Weekend', 'Lag_1_Power',
        'Lag_2_Power', 'Lag_3_Power', 'Temperature', 'Humidity', 'Num_Fridges', 
        'Num_Washing_Machines', 'Num_ACs', 'Num_Fans', 'Num_TVs', 'Power_Fridges', 
        'Power_Washing_Machines', 'Power_ACs', 'Power_Fans', 'Power_TVs', 'Voltage', 
        'Global_intensity', 'Hour_sin', 'Hour_cos', 'Day_of_Week_sin', 'Day_of_Week_cos', 
        'Rolling_Mean_Power', 'Rolling_Mean_Power_6', 'Power_Diff', 'Voltage_Diff',
        'Temp_Humidity_Ratio'
    ]

    return data, features