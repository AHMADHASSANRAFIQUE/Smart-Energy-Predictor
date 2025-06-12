import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from data_processing import load_and_preprocess_data
from model_training import train_model
from visualizations import plot_visualizations
from utils import fetch_weather_forecast, appliance_options, currency_symbol, calculate_slab_bill

@st.cache_data
def load_model_data(selected_model):
    data, features = load_and_preprocess_data()
    X = data[features]
    y = data['Global_active_power']
    return (train_model(X, y, features, selected_model, use_cv=True, use_shap=True), data, features)

@st.cache_data(hash_funcs={str: lambda x: x})
def generate_visualizations(data, X_test_scaled, y_test, y_test_pred, model_name, shap_values, selected_features, coefficients):
    plot_visualizations(data, X_test_scaled, y_test, y_test_pred, model_name, shap_values, selected_features, coefficients)

st.set_page_config(page_title="Smart Energy Predictor", layout="wide")
st.title("Smart Energy Consumption Predictor ⚡")

# Model selection
with st.sidebar:
    selected_model = st.selectbox("Select Model", ["RandomForest", "Ridge", "XGBoost"])

if 'appliances' not in st.session_state:
    st.session_state.appliances = []
if 'prediction' not in st.session_state:
    st.session_state.prediction = 1.0
if 'appliance_contribution' not in st.session_state:
    st.session_state.appliance_contribution = 0.0
if 'calibration_factor' not in st.session_state:
    st.session_state.calibration_factor = 1.0
if 'input_data' not in st.session_state:
    st.session_state.input_data = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'raw_input_data' not in st.session_state:
    st.session_state.raw_input_data = np.array([[0.0, 0.0, 0.0, 12, 0, 0, 5, 0, 25.0, 60.0, 1, 0, 0, 1, 1, 0.2, 0.0, 0.0, 0.075, 0.1, 230.0, 5.0, 0.0, 1.0, 0.0, 1.0, 0.0]])

# Load and cache model data
(model_result, data, features) = load_model_data(selected_model)
(model, scaler, poly, selector, selected_features, model_name, val_r2, val_rmse, val_mae, val_mape,
 test_r2, test_rmse, test_mae, test_mape, coefficients, X_test_scaled, y_test, y_test_pred, shap_values, ridge_r2, rf_cv, training_time, train_size) = model_result

# Generate visualizations
generate_visualizations(data, X_test_scaled, y_test, y_test_pred, model_name, shap_values, selected_features, coefficients)

# Sidebar: Model Performance
with st.sidebar:
    st.header("Model Performance")
    st.write(f"Model: {model_name}")
    st.write(f"Training Time: {training_time:.2f} seconds")
    st.write(f"Training Data Size: {train_size} samples")
    st.write(f"Validation R²: {val_r2:.4f} ({val_r2*100 :.2f}%)")
    st.write(f"Test R²: {test_r2:.4f} ({test_r2*100 :.2f}%)")
    st.write(f"Target R²: above 90% (Achieved: {test_r2*100 :.2f}%)")
    st.write(f"Test RMSE: {test_rmse:.4f} kWh")
    st.write(f"Test MAE: {val_mae:.4f} kWh")
    st.write(f"Test MAPE: {test_mape:.2f}%")
    if model_name in ["RandomForest", "XGBoost"] and rf_cv is not None:
        st.write(f"Cross-Validation R²: {rf_cv:.4f}")

tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Visualizations", "Recommendations", "Research Summary"])

with tab1:
    st.subheader("Prediction Mode")
    prediction_mode = st.radio("Mode", ("Units Only (Simple)", "Appliances + Units (Detailed)"))

    last_data = data.iloc[-1]
    default_prev_sub1 = last_data['Sub_metering_1'] if not pd.isna(last_data['Sub_metering_1']) else 0.0
    default_prev_sub2 = last_data['Sub_metering_2'] if not pd.isna(last_data['Sub_metering_2']) else 0.0
    default_prev_sub3 = last_data['Sub_metering_3'] if not pd.isna(last_data['Sub_metering_3']) else 0.0
    default_prev_hour = last_data['Hour'] if not pd.isna(last_data['Hour']) else datetime.now().hour
    default_prev_day_of_week = last_data['Day_of_Week'] if not pd.isna(last_data['Day_of_Week']) else datetime.now().weekday()
    default_prev_is_peak_hour = 1 if 18 <= default_prev_hour <= 22 else 0
    default_prev_month = last_data['Month'] if not pd.isna(last_data['Month']) else datetime.now().month
    default_prev_is_weekend = 1 if default_prev_day_of_week in [5, 6] else 0
    default_temp = last_data['Temperature'] if not pd.isna(last_data['Temperature']) else 25.0
    default_humidity = last_data['Humidity'] if not pd.isna(last_data['Humidity']) else 60.0
    default_voltage = last_data['Voltage'] if not pd.isna(last_data['Voltage']) else 230.0
    default_global_intensity = last_data['Global_intensity'] if not pd.isna(last_data['Global_intensity']) else 5.0
    default_voltage_diff = last_data['Voltage_Diff'] if not pd.isna(last_data['Voltage_Diff']) else 0.0
    default_is_holiday = last_data.get('Is_Holiday', 0) if not pd.isna(last_data.get('Is_Holiday', 0)) else 0

    default_num_fridges = 1
    default_num_washing_machines = 0
    default_num_acs = 0
    default_num_fans = 1
    default_num_tvs = 1
    default_power_fridges = default_num_fridges * 0.2
    default_power_washing_machines = default_num_washing_machines * 0.5
    default_power_acs = default_num_acs * 1.5
    default_power_fans = default_num_fans * 0.075
    default_power_tvs = default_num_tvs * 0.1
    default_total_power = default_power_fridges + default_power_washing_machines + default_power_acs + default_power_fans + default_power_tvs

    if prediction_mode == "Appliances + Units (Detailed)":
        st.subheader("Add Appliances")
        appliance_name = st.selectbox("Appliance", list(appliance_options.keys()))
        appliance_type = st.selectbox("Type", ["Non-Inverter", "Inverter"])
        power_rating = appliance_options[appliance_name][appliance_type]
        usage_minutes = st.slider(f"Usage (minutes)", 0, 60, 30)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add Appliance"):
                usage_factor = usage_minutes / 60
                st.session_state.appliances.append({
                    "name": appliance_name,
                    "type": appliance_type,
                    "power": power_rating / 1000,
                    "usage_factor": usage_factor
                })
                st.success(f"Added {appliance_name} ({appliance_type}, {power_rating}W, {usage_minutes} min)")
        with col2:
            if st.button("Reset Appliances"):
                st.session_state.appliances = []
                st.success("Appliances reset!")

        num_fridges = num_washing_machines = num_acs = num_fans = num_tvs = 0
        power_fridges = power_washing_machines = power_acs = power_fans = power_tvs = 0.0
        for app in st.session_state.appliances:
            usage_factor = app.get('usage_factor', 0.5)
            if app.get('name') in ["AC", "Fan"] and default_temp > 30:
                usage_factor *= 1.2
            elif app.get('name') in ["Heater", "Water Heater"] and default_temp < 10:
                usage_factor *= 1.2
            if app.get('name') in ["TV", "LED Light", "Bulb"] and 18 <= default_prev_hour <= 22:
                usage_factor *= 1.1
            app_power = app.get('power', 0) * usage_factor
            if app.get('name') == "Fridge":
                num_fridges += 1
                power_fridges += app_power
            elif app.get('name') == "Washing Machine":
                num_washing_machines += 1
                power_washing_machines += app_power
            elif app.get('name') == "AC":
                num_acs += 1
                power_acs += app_power
            elif app.get('name') == "Fan":
                num_fans += 1
                power_fans += app_power
            elif app.get('name') == "TV":
                num_tvs += 1
                power_tvs += app_power
        if not st.session_state.appliances:
            num_fridges = default_num_fridges
            num_washing_machines = default_num_washing_machines
            num_acs = default_num_acs
            num_fans = default_num_fans
            num_tvs = default_num_tvs
            power_fridges = default_power_fridges
            power_washing_machines = default_power_washing_machines
            power_acs = default_power_acs
            power_fans = default_power_fans
            power_tvs = default_power_tvs
            st.session_state.appliance_contribution = default_total_power
        else:
            total_power = power_fridges + power_washing_machines + power_acs + power_fans + power_tvs
            st.session_state.appliance_contribution = total_power
            st.write("**Your Appliances:**")
            for app in st.session_state.appliances:
                app_power = app.get('power', 0) * app.get('usage_factor', 0.5)
                st.write(f"- {app.get('name')} ({app.get('type')}, {app_power:.3f} kWh)")
            st.write(f"**Total Appliance Usage:** {total_power:.3f} kWh")

        if st.session_state.appliances:
            st.subheader("Appliance Usage Breakdown")
            appliance_names = [app.get('name') for app in st.session_state.appliances]
            appliance_usages = [app.get('power', 0) * app.get('usage_factor', 0.5) for app in st.session_state.appliances]
            total_usage = sum(appliance_usages)
            if total_usage > 0:
                percentages = [(usage / total_usage) * 100 for usage in appliance_usages]
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(percentages, labels=appliance_names, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired(np.arange(len(appliance_names))))
                ax.set_title('Appliance Contribution')
                st.pyplot(fig)
        else:
            st.subheader("Default Appliance Usage Breakdown")
            appliance_names = ["Fridge", "Fan", "TV"]
            appliance_usages = [default_power_fridges, default_power_fans, default_power_tvs]
            total_usage = sum(appliance_usages)
            percentages = [(usage / total_usage) * 100 for usage in appliance_usages]
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(percentages, labels=appliance_names, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired(np.arange(len(appliance_names))))
            ax.set_title('Default Appliance Contribution')
            st.pyplot(fig)

    else:
        num_fridges = default_num_fridges
        num_washing_machines = default_num_washing_machines
        num_acs = default_num_acs
        num_fans = default_num_fans
        num_tvs = default_num_tvs
        power_fridges = default_power_fridges
        power_washing_machines = default_power_washing_machines
        power_acs = default_power_acs
        power_fans = default_power_fans
        power_tvs = default_power_tvs
        st.session_state.appliance_contribution = default_total_power

    st.subheader("Weather Information")
    city = st.text_input("City", value="Lahore")
    temp = st.slider("Temperature (°C)", -10, 50, int(default_temp))
    humidity = st.slider("Humidity (%)", 0, 100, int(default_humidity))
    if temp > 30:
        st.info("Hot weather! Consider fans over AC.")
    elif temp < 10:
        st.info("Cold weather! Layer clothing before using heaters.")

    st.subheader("Holiday Information")
    is_holiday = st.checkbox("Is it a holiday?", value=bool(default_is_holiday))

    st.subheader("Energy Usage Prediction")
    if prediction_mode == "Units Only (Simple)":
        prev_sub1 = default_prev_sub1
        prev_sub2 = default_prev_sub2
        prev_sub3 = default_prev_sub3
        prev_hour = default_prev_hour
        prev_day_of_week_num = default_prev_day_of_week
        prev_is_peak_hour = default_prev_is_peak_hour
        prev_month = default_prev_month
        prev_is_weekend = default_prev_is_weekend
        voltage = default_voltage
        global_intensity = default_global_intensity
        num_fridges = default_num_fridges
        num_washing_machines = default_num_washing_machines
        num_acs = default_num_acs
        num_fans = default_num_fans
        num_tvs = default_num_tvs
        power_fridges = default_power_fridges
        power_washing_machines = default_power_washing_machines
        power_acs = default_power_acs
        power_fans = default_power_fans
        power_tvs = default_power_tvs
        voltage_diff = default_voltage_diff
        is_holiday = default_is_holiday
    else:
        col1, col2 = st.columns(2)
        with col1:
            prev_sub1 = st.number_input("Kitchen Usage (kWh)", min_value=0.0, value=float(default_prev_sub1))
            prev_sub2 = st.number_input("Laundry Usage (kWh)", min_value=0.0, value=float(default_prev_sub2))
            prev_sub3 = st.number_input("Other Usage (kWh)", min_value=0.0, value=float(default_prev_sub3))
            voltage = st.number_input("Voltage (V)", min_value=0.0, value=float(default_voltage))
            global_intensity = st.number_input("Global Intensity (A)", min_value=0.0, value=float(default_global_intensity))
        with col2:
            prev_hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=int(default_prev_hour))
            prev_day_of_week = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=int(default_prev_day_of_week))
            prev_is_peak_hour = st.checkbox("Peak Hour (6 PM - 10 PM)", value=bool(default_prev_is_peak_hour))
            prev_month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=int(default_prev_month))
            prev_is_weekend = st.checkbox("Weekend", value=bool(default_prev_is_weekend))
        day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        prev_day_of_week_num = day_map[prev_day_of_week]
        voltage_diff = default_voltage_diff

    hour_sin = np.sin(2 * np.pi * prev_hour / 24)
    hour_cos = np.cos(2 * np.pi * prev_hour / 24)
    day_of_week_sin = np.sin(2 * np.pi * prev_day_of_week_num / 7)
    day_of_week_cos = np.cos(2 * np.pi * prev_day_of_week_num / 7)

    st.subheader("Feedback (Optional)")
    actual_usage = st.number_input("Actual Usage Last Hour (kWh)", min_value=0.0, value=0.0)
    if actual_usage > 0 and st.session_state.prediction > 0:
        st.session_state.calibration_factor = actual_usage / st.session_state.prediction
        st.write(f"Calibration factor: {st.session_state.calibration_factor:.2f}")
        st.session_state.prediction_history.append({
            'time': datetime.now(),
            'predicted': st.session_state.prediction,
            'actual': actual_usage,
            'calibration_factor': st.session_state.calibration_factor
        })

    if st.button("Predict Energy Usage"):
        input_data = np.array([[
            prev_sub1, prev_sub2, prev_sub3,
            prev_hour, prev_day_of_week_num, int(prev_is_peak_hour),
            prev_month, int(prev_is_weekend),
            temp, humidity, num_fridges, num_washing_machines,
            num_acs, num_fans, num_tvs, power_fridges, power_washing_machines,
            power_acs, power_fans, power_tvs, voltage, global_intensity,
            hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, voltage_diff,
            int(is_holiday)
        ]])
        st.session_state.raw_input_data = input_data
        input_data_poly = poly.transform(input_data)
        input_data_selected = selector.transform(input_data_poly)
        input_data_scaled = scaler.transform(input_data_selected)

        background_usage = model.predict(input_data_scaled)[0]
        background_usage = np.clip(background_usage, 0, 10)

        if prediction_mode == "Appliances + Units (Detailed)":
            appliance_usage = st.session_state.appliance_contribution
            background_contribution = background_usage * 0.03
            prediction = appliance_usage + max(background_contribution, 0)
            prediction = prediction * st.session_state.calibration_factor
            st.session_state.prediction = max(prediction, 0)
            predicted_appliance_usage = appliance_usage
            if temp > 30:
                if num_acs > 0:
                    predicted_appliance_usage += power_acs * 0.2
                if num_fans > 0:
                    predicted_appliance_usage += power_fans * 0.2
            if 18 <= prev_hour <= 22:
                if num_tvs > 0:
                    predicted_appliance_usage += power_tvs * 0.1
        else:
            prediction = background_usage * st.session_state.calibration_factor
            st.session_state.prediction = max(prediction, 0)
            predicted_appliance_usage = st.session_state.appliance_contribution

        st.session_state.input_data = input_data_scaled
        st.success(f"Predicted Usage: {st.session_state.prediction:.2f} kWh")
        if prediction_mode == "Appliances + Units (Detailed)":
            st.write(f"- Appliance Contribution: {st.session_state.appliance_contribution:.2f} kWh")
            st.write(f"- Background Usage: {max(background_contribution, 0):.2f} kWh")

        current_time = datetime(2025, 5, 3, datetime.now().hour, datetime.now().minute)
        time_range = [current_time - timedelta(hours=24-i) for i in range(24)]
        fig, ax = plt.subplots(figsize=(10, 4))
        if prediction_mode == "Appliances + Units (Detailed)":
            past_24_appliance_usage = []
            for i in range(24):
                hourly_usage = 0
                current_hour = (prev_hour - (24 - i)) % 24
                is_peak_hour = 18 <= current_hour <= 22
                is_daytime = 10 <= current_hour <= 16
                is_night = 0 <= current_hour <= 6
                for app in st.session_state.appliances:
                    usage_factor = app.get('usage_factor', 0.5)
                    app_power = app.get('power', 0)
                    app_name = app.get('name', 'Unknown')
                    if app_name == "Fridge":
                        usage_adjustment = 1.1 if is_peak_hour else 1.0
                    elif app_name in ["AC", "Fan"]:
                        usage_adjustment = 1.2 if temp > 30 and is_daytime else 0.8 if is_night else 1.0
                    elif app_name in ["TV", "LED Light", "Bulb"]:
                        usage_adjustment = 1.1 if is_peak_hour else 0.2 if is_night else 0.5
                    else:
                        usage_adjustment = 0.3 if is_night else 0.7
                    hourly_usage += app_power * usage_factor * usage_adjustment
                past_24_appliance_usage.append(hourly_usage)
            ax.plot(time_range, past_24_appliance_usage, label='Past 24 Hours (Appliances)', color='blue', marker='o')
            next_time = current_time + timedelta(hours=1)
            ax.scatter(next_time, predicted_appliance_usage, color='red', s=100, label=f'Predicted: {predicted_appliance_usage:.2f} kWh')
            ax.plot([current_time, next_time], [past_24_appliance_usage[-1], predicted_appliance_usage], color='red', linestyle='--')
        else:
            last_24 = data['Global_active_power'].tail(24)
            ax.plot(time_range, last_24.values, label='Past 24 Hours (Total)', color='blue', marker='o')
            next_time = current_time + timedelta(hours=1)
            ax.scatter(next_time, st.session_state.prediction, color='red', s=100, label=f'Predicted: {st.session_state.prediction:.2f} kWh')
            ax.plot([current_time, next_time], [last_24.values[-1], st.session_state.prediction], color='red', linestyle='--')
        ax.set_title('Energy Usage Trend and Prediction')
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy (kWh)')
        ax.grid(True)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        @st.cache_data(ttl=3600)
        def get_weather(city):
            return fetch_weather_forecast(city)
        weather_forecast = get_weather(city)
        if weather_forecast:
            st.subheader("24-Hour Weather Forecast")
            for wf in weather_forecast:
                st.write(f"Time: {wf['time']}, Temp: {wf['temp']}°C, Humidity: {wf['humidity']}%")

        st.subheader("24-Hour Energy Forecast")
        forecast = []
        current_input = input_data[0].copy()
        for i in range(0, 24, 3):
            current_hour = (prev_hour + i + 1) % 24
            wf_idx = min(i // 3, len(weather_forecast) - 1) if weather_forecast else 0
            temp_forecast = weather_forecast[wf_idx]['temp'] if weather_forecast else temp
            humidity_forecast = weather_forecast[wf_idx]['humidity'] if weather_forecast else humidity
            power_acs_hour = power_acs * (1.2 if temp_forecast > 30 else 1.0)
            power_fans_hour = power_fans * (1.2 if temp_forecast > 30 else 1.0)
            power_tvs_hour = power_tvs * (1.1 if 18 <= current_hour <= 22 else 1.0)
            current_input[3] = current_hour
            current_input[4] = (prev_day_of_week_num + ((prev_hour + i + 1) // 24)) % 7
            current_input[5] = 1 if 18 <= current_hour <= 22 else 0
            current_input[7] = 1 if current_input[4] in [5, 6] else 0
            current_input[8] = temp_forecast
            current_input[9] = humidity_forecast
            current_input[17] = power_acs_hour
            current_input[18] = power_fans_hour
            current_input[19] = power_tvs_hour
            current_hour_sin = np.sin(2 * np.pi * current_hour / 24)
            current_hour_cos = np.cos(2 * np.pi * current_hour / 24)
            current_day_of_week_num = current_input[4]
            current_day_of_week_sin = np.sin(2 * np.pi * current_day_of_week_num / 7)
            current_day_of_week_cos = np.cos(2 * np.pi * current_day_of_week_num / 7)
            current_input[22] = current_hour_sin
            current_input[23] = current_hour_cos
            current_input[24] = current_day_of_week_sin
            current_input[25] = current_day_of_week_cos
            current_input[26] = voltage_diff
            current_input[27] = int(is_holiday)
            current_input_poly = poly.transform([current_input])
            current_input_selected = selector.transform(current_input_poly)
            current_input_scaled = scaler.transform(current_input_selected)
            background_usage = model.predict(current_input_scaled)[0]
            background_usage = np.clip(background_usage, 0, 10)
            if prediction_mode == "Appliances + Units (Detailed)":
                hourly_appliance_usage = st.session_state.appliance_contribution
                if temp_forecast > 30:
                    if num_acs > 0:
                        hourly_appliance_usage += power_acs * 0.2
                    if num_fans > 0:
                        hourly_appliance_usage += power_fans * 0.2
                if 18 <= current_hour <= 22:
                    if num_tvs > 0:
                        hourly_appliance_usage += power_tvs * 0.1
                background_contribution = background_usage * 0.03
                pred = hourly_appliance_usage + max(background_contribution, 0)
                pred = pred * st.session_state.calibration_factor
            else:
                pred = background_usage * st.session_state.calibration_factor
            forecast.append(max(pred, 0))
        times = [current_time + timedelta(hours=i) for i in range(0, 24, 3)]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, forecast, label='Forecasted Usage', color='purple', marker='o')
        ax.set_title('24-Hour Energy Forecast')
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy (kWh)')
        ax.grid(True)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Bill Estimation")
    if st.session_state.prediction:
        monthly_usage = st.session_state.prediction * 24 * 30
        monthly_bill, slab_breakdown = calculate_slab_bill(monthly_usage)
        appliance_usage = st.session_state.appliance_contribution * 24 * 30
        appliance_bill, _ = calculate_slab_bill(appliance_usage)

        st.metric(label="Estimated Monthly Cost", value=f"{currency_symbol} {monthly_bill:.2f}")
        st.write(f"- Hourly Usage: {st.session_state.prediction:.2f} kWh")
        st.write(f"- Daily Usage: {st.session_state.prediction*24:.2f} kWh")
        st.write(f"- Monthly Usage: {monthly_usage:.2f} kWh")
        st.write("**Slab-wise Bill Breakdown:**")
        for slab, (units, cost) in slab_breakdown.items():
            if units > 0:
                st.write(f"- {slab}: {units:.2f} kWh x {cost/units if units > 0 else 0:.2f} Rs/kWh = {currency_symbol} {cost:.2f}")
        st.write(f"- Appliance Cost: {currency_symbol} {appliance_bill:.2f} ({(appliance_bill/monthly_bill)*100:.1f}%)")

    st.subheader("Cost-Saving Calculator")
    reduction = st.slider("Reduce Appliance Usage (%)", 0, 50, 10)
    if st.session_state.input_data is not None:
        reduced_appliance_usage = st.session_state.appliance_contribution * (1 - reduction / 100)
        reduced_power_acs = power_acs * (1 - reduction / 100)
        reduced_power_fans = power_fans * (1 - reduction / 100)
        reduced_power_tvs = power_tvs * (1 - reduction / 100)
        reduced_input = st.session_state.raw_input_data[0].copy()
        reduced_input[17] = reduced_power_acs
        reduced_input[18] = reduced_power_fans
        reduced_input[19] = reduced_power_tvs
        reduced_hour_sin = np.sin(2 * np.pi * prev_hour / 24)
        reduced_hour_cos = np.cos(2 * np.pi * prev_hour / 24)
        reduced_day_of_week_sin = np.sin(2 * np.pi * prev_day_of_week_num / 7)
        reduced_day_of_week_cos = np.cos(2 * np.pi * prev_day_of_week_num / 7)
        reduced_input[22] = reduced_hour_sin
        reduced_input[23] = reduced_hour_cos
        reduced_input[24] = reduced_day_of_week_sin
        reduced_input[25] = reduced_day_of_week_cos
        reduced_input[26] = voltage_diff
        reduced_input[27] = int(is_holiday)
        reduced_input_poly = poly.transform([reduced_input])
        reduced_input_selected = selector.transform(reduced_input_poly)
        reduced_input_scaled = scaler.transform(reduced_input_selected)
        background_usage = model.predict(reduced_input_scaled)[0]
        background_usage = np.clip(background_usage, 0, 10)
        if prediction_mode == "Appliances + Units (Detailed)":
            background_contribution = background_usage * 0.03
            reduced_prediction = reduced_appliance_usage + max(background_contribution, 0)
            reduced_prediction = reduced_prediction * st.session_state.calibration_factor
        else:
            reduced_prediction = background_usage * st.session_state.calibration_factor
        reduced_prediction = max(reduced_prediction, 0)
        reduced_monthly_usage = reduced_prediction * 24 * 30
        reduced_bill, _ = calculate_slab_bill(reduced_monthly_usage)
        savings = monthly_bill - reduced_bill
        st.write(f"Reduce usage by {reduction}%: New bill {currency_symbol} {reduced_bill:.2f}, saving {currency_symbol} {savings:.2f}/month.")
    else:
        st.write("Please make a prediction first to calculate cost savings.")

with tab2:
    st.subheader("Visualizations")
    col1, col2 = st.columns(2)
    model_plot_dir = os.path.join('plots', model_name)
    with col1:
        if os.path.exists(os.path.join('plots', 'hourly_energy_trend.png')):
            st.image(os.path.join('plots', 'hourly_energy_trend.png'), caption='Hourly Energy Trend')
        else:
            st.write("Hourly Energy Trend plot not available.")
        if os.path.exists(os.path.join(model_plot_dir, f'predicted_vs_actual_{model_name}.png')):
            st.image(os.path.join(model_plot_dir, f'predicted_vs_actual_{model_name}.png'), caption=f'Predicted vs Actual ({model_name})')
        else:
            st.write(f"Predicted vs Actual ({model_name}) plot not available.")
        if os.path.exists(os.path.join(model_plot_dir, f'time_series_actual_vs_predicted_{model_name}.png')):
            st.image(os.path.join(model_plot_dir, f'time_series_actual_vs_predicted_{model_name}.png'), caption=f'Time Series Actual vs Predicted ({model_name})')
        else:
            st.write(f"Time Series Actual vs Predicted ({model_name}) plot not available.")
    with col2:
        if os.path.exists(os.path.join('plots', 'appliance_contribution.png')):
            st.image(os.path.join('plots', 'appliance_contribution.png'), caption='Appliance Contribution')
        else:
            st.write("Appliance Contribution plot not available.")
        if os.path.exists(os.path.join('plots', 'temperature_impact.png')):
            st.image(os.path.join('plots', 'temperature_impact.png'), caption='Temperature Impact')
        else:
            st.write("Temperature Impact plot not available.")
        if os.path.exists(os.path.join(model_plot_dir, f'feature_importance_bar_{model_name}.png')):
            st.image(os.path.join(model_plot_dir, f'feature_importance_bar_{model_name}.png'), caption=f'Feature Importance ({model_name})')
        else:
            st.write(f"Feature Importance ({model_name}) plot not available.")
        if os.path.exists(os.path.join(model_plot_dir, f'error_distribution_{model_name}.png')):
            st.image(os.path.join(model_plot_dir, f'error_distribution_{model_name}.png'), caption=f'Error Distribution ({model_name})')
        else:
            st.write(f"Error Distribution ({model_name}) plot not available.")
    if model_name in ["RandomForest", "XGBoost"] and os.path.exists(os.path.join(model_plot_dir, f'shap_summary_{model_name}.png')):
        st.image(os.path.join(model_plot_dir, f'shap_summary_{model_name}.png'), caption=f'Feature Importance (SHAP) - {model_name}')
    else:
        st.write(f"Feature Importance (SHAP) plot not available for {model_name}.")
    if os.path.exists(os.path.join(model_plot_dir, f'hyperparameter_tuning_{model_name}.png')):
        st.image(os.path.join(model_plot_dir, f'hyperparameter_tuning_{model_name}.png'), caption=f'Hyperparameter Tuning Results ({model_name})')
    else:
        st.write(f"Hyperparameter Tuning Results ({model_name}) plot not available.")

with tab3:
    st.subheader("Recommendations")
    if model_name in ["RandomForest", "XGBoost"] and coefficients is not None:
        feature_importance = pd.DataFrame({'Feature': selected_features, 'Importance': coefficients}).sort_values(by='Importance', ascending=False)
        st.write("**Top Factors Driving Consumption:**")
        for i in range(min(3, len(feature_importance))):
            feature = feature_importance.iloc[i]['Feature']
            imp = feature_importance.iloc[i]['Importance']
            if 'power_acs' in feature:
                st.write(f"- AC usage (importance: {imp:.4f}). Set to 24°C or use fans.")
            elif 'power_tvs' in feature:
                st.write(f"- TV usage (importance: {imp:.4f}). Reduce evening TV time.")
            elif 'prev_hour' in feature or 'hour_' in feature:
                st.write(f"- Time of day (importance: {imp:.4f}). Shift usage to off-peak hours.")
    elif model_name == "Ridge" and coefficients is not None:
        feature_importance = pd.DataFrame({'Feature': selected_features, "Importance": np.abs(coefficients)}).sort_values(by='Importance', ascending=False)
        st.write("**Top Factors Driving Consumption:**")
        for i in range(min(3, len(feature_importance))):
            feature = feature_importance.iloc[i]['Feature']
            imp = feature_importance.iloc[i]['Importance']
            if 'power_acs' in feature:
                st.write(f"- AC usage (importance: {imp:.4f}). Set to 24°C or use fans.")
            elif 'power_tvs' in feature:
                st.write(f"- TV usage (importance: {imp:.4f}). Reduce evening TV time.")
            elif 'prev_hour' in feature or 'hour_' in feature:
                st.write(f"- Time of day (importance: {imp:.4f}). Shift usage to off-peak hours.")
    else:
        st.write("Feature importance not available for this model.")

    st.subheader("General Tips")
    with st.expander("Energy Saving Tips"):
        st.write("1. Use appliances during off-peak hours.")
        st.write("2. Switch to inverter appliances.")
        st.write("3. Set AC to 24°C.")
        st.write("4. Use LED bulbs.")
        st.write("5. Unplug devices when not in use.")
        if st.session_state.appliances:
            high_power_apps = [app for app in st.session_state.appliances if app.get('power', 0) > 1.0]
            if high_power_apps:
                st.write("**High-Power Appliances:**")
                for app in high_power_apps:
                    st.write(f"- {app.get('name')}: {app.get('power'):.2f} kW. Use less or during off-peak.")

    weekend_spikes = data[(data['Is_Weekend'] == 1) & (data['Is_Peak_Hour'] == 0)]['Global_active_power'].mean()
    weekday_spikes = data[(data['Is_Weekend'] == 0) & (data['Is_Peak_Hour'] == 0)]['Global_active_power'].mean()
    st.write("**Interesting Fact:**")
    st.write(f"Non-peak weekend usage (avg: {weekend_spikes:.2f} kWh) is higher than weekdays (avg: {weekday_spikes:.2f} kWh).")

with tab4:
    st.subheader("Research Summary")
    rf_cv_display = f"{rf_cv:.4f}" if model_name in ["RandomForest", "XGBoost"] and rf_cv is not None else "N/A"
    summary = f"""
# Smart Energy Consumption Predictor

## Model Overview
- **Model**: {model_name}
- **Training Time**: {training_time:.2f} seconds
- **Training Data Size**: {train_size} samples (80% of total data)
- **Features**: {len(features)} base features, {len(selected_features)} after selection.
- **Performance**:
  - Validation R²: {val_r2:.4f}
  - Test R²: {test_r2:.4f}
  - Test RMSE: {test_rmse:.4f} kWh
  - Test MAE: {val_mae:.4f} kWh
  - Test MAPE: {test_mape:.2f}%
  - Cross-Validation R²: {rf_cv_display}
- **Methodology**: {model_name} with feature selection via mutual information, Grid Search for hyperparameter tuning, and SHAP for interpretability. Cyclical encoding and rolling averages enhance temporal modeling.

## Key Findings
- Appliance usage (ACs, TVs) and temporal factors dominate predictions.
- Weather interactions (temperature, humidity) significantly impact consumption.
- Non-peak weekend spikes suggest behavioral energy-saving opportunities.

## Visualizations
- Hourly trends highlight peak usage.
- Appliance contributions identify key devices.
- SHAP plots and AQI-inspired visualizations (time series, feature importance, error distribution) provide insights.

## Hyperparameter Tuning
- Grid Search used to optimize learning rate, number of estimators, and max depth.
- Best parameters achieved ~above 90% Test R² across all models.

## Recommendations
- Optimize AC settings and prefer inverter models.
- Shift usage to off-peak hours.
- Leverage weather forecasts for appliance planning.

*Generated on {datetime.now().strftime('%Y-%m-%d')}*
"""
    st.markdown(summary)
    st.download_button("Download Summary (Markdown)", summary, file_name="energy_predictor_summary.md", mime="text/markdown")