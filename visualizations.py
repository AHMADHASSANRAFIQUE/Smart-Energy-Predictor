import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
import pandas as pd

def plot_visualizations(data, X_test_scaled, y_test, y_test_pred, model_name, shap_values, selected_features, coefficients):
    """Generate model-specific visualizations including AQI-inspired and hyperparameter tuning plots."""    
    model_plot_dir = os.path.join('plots', model_name)
    if not os.path.exists(model_plot_dir):
        os.makedirs(model_plot_dir)

    # Hourly Energy Trend
    if 'Hour' in data.columns and 'Global_active_power' in data.columns:
        hourly_trend = data.groupby('Hour')['Global_active_power'].mean()
        plt.figure(figsize=(10, 6))
        hourly_trend.plot(kind='line', marker='o', color='blue')
        plt.title('Average Hourly Energy Consumption', fontsize=14)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Energy (kWh)', fontsize=12)
        plt.grid(True)
        plt.savefig(os.path.join('plots', 'hourly_energy_trend.png'))
        plt.close()

    # Predicted vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'Predicted vs Actual Energy Consumption ({model_name})', fontsize=14)
    plt.xlabel('Actual (kWh)', fontsize=12)
    plt.ylabel('Predicted (kWh)', fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(model_plot_dir, f'predicted_vs_actual_{model_name}.png'))
    plt.close()

    # Appliance Contribution
    appliance_cols = ['Power_Fridges', 'Power_Washing_Machines', 'Power_ACs', 'Power_Fans', 'Power_TVs']
    if all(col in data.columns for col in appliance_cols):
        appliance_data = data[appliance_cols].mean()
        plt.figure(figsize=(8, 8))
        appliance_data.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired(np.arange(len(appliance_data))))
        plt.title('Appliance Contribution to Energy Usage', fontsize=14)
        plt.ylabel('')
        plt.savefig(os.path.join('plots', 'appliance_contribution.png'))
        plt.close()

    # Temperature Impact
    if 'Temperature' in data.columns and 'Global_active_power' in data.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Temperature', y='Global_active_power', data=data, alpha=0.5, color='orange')
        plt.title('Temperature vs Energy Consumption', fontsize=14)
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Energy (kWh)', fontsize=12)
        plt.grid(True)
        plt.savefig(os.path.join('plots', 'temperature_impact.png'))
        plt.close()

    if model_name in ["RandomForest", "XGBoost"] and shap_values is not None:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_scaled, feature_names=selected_features, show=False)
        plt.title(f'Feature Importance (SHAP) - {model_name}', fontsize=14)
        plt.savefig(os.path.join(model_plot_dir, f'shap_summary_{model_name}.png'))
        plt.close()

    # Bar Plot
    if model_name in ["RandomForest", "XGBoost"] and shap_values is not None:
        feature_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title(f'Top 10 Feature Importance ({model_name})', fontsize=14)
        plt.xlabel('Mean SHAP Value', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.grid(True, axis='x')
        plt.savefig(os.path.join(model_plot_dir, f'feature_importance_bar_{model_name}.png'))
        plt.close()
    elif model_name == "Ridge" and coefficients is not None:
        feature_importance = np.abs(coefficients)
        importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title(f'Top 10 Feature Importance (Ridge Coefficients)', fontsize=14)
        plt.xlabel('Absolute Coefficient Value', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.grid(True, axis='x')
        plt.savefig(os.path.join(model_plot_dir, f'feature_importance_bar_{model_name}.png'))
        plt.close()

    # Actual vs Predicted
    if 'Date' in data.columns:
        plt.figure(figsize=(12, 6))
        test_dates = data['Date'].iloc[-len(y_test):].values
        plt.plot(test_dates, y_test, label='Actual Energy (kWh)', color='blue', marker='o', alpha=0.7)
        plt.plot(test_dates, y_test_pred, label='Predicted Energy (kWh)', color='red', linestyle='--', marker='x', alpha=0.7)
        plt.title(f'Actual vs Predicted Energy Consumption Over Time ({model_name})', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Energy (kWh)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(model_plot_dir, f'time_series_actual_vs_predicted_{model_name}.png'))
        plt.close()

    # Error
    errors = y_test - y_test_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=30, kde=True, color='purple')
    plt.title(f'Distribution of Prediction Errors ({model_name})', fontsize=14)
    plt.xlabel('Error (kWh)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(model_plot_dir, f'error_distribution_{model_name}.png'))
    plt.close()

    # Hyperparameter
    log_file = 'experiment_log.csv'
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        if model_name in log_df['model_type'].values:
            model_log = log_df[log_df['model_type'] == model_name]
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=model_log.index, y='test_r2', data=model_log, marker='o', label='Test R²')
            plt.title(f'Hyperparameter Tuning Results ({model_name})', fontsize=14)
            plt.xlabel('Experiment', fontsize=12)
            plt.ylabel('Test R²', fontsize=12)
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(model_plot_dir, f'hyperparameter_tuning_{model_name}.png'))
            plt.close()