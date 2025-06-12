import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import shap
import time
import os
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression

def train_model(X, y, features, model_type="RandomForest", use_cv=True, use_shap=True):    
    print(f"Starting training for {model_type}...")
    missing_features = [f for f in features if f not in X.columns]
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")
    
    exclude_features = ['Global_active_power', 'Lag_1_Power', 'Lag_2_Power', 'Lag_3_Power', 
                       'Rolling_Mean_Power', 'Rolling_Mean_Power_6', 'Power_Diff']
    features = [f for f in features if f not in exclude_features]
    print(f"Using features: {features}")
    
    train_size = int(0.8 * len(X))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    total_train_size = len(X_train) + len(X_val)
    
    np.random.seed(42)
    noise = np.random.normal(0, 0.05 * y_train.std(), size=len(y_train))
    y_train_noisy = y_train + noise

    print("Preprocessing features...")
    start_preprocess = time.time()
    X_train_scaled, selected_features, poly, scaler, selector = preprocess_features(X_train, y_train_noisy, features)
    X_val_poly = poly.transform(X_val[features])
    X_val_selected = selector.transform(X_val_poly)
    X_val_scaled = scaler.transform(X_val_selected)
    X_test_poly = poly.transform(X_test[features])
    X_test_selected = selector.transform(X_test_poly)
    X_test_scaled = scaler.transform(X_test_selected)
    print(f"Preprocessing completed in {time.time() - start_preprocess:.2f} seconds")

    start_time = time.time()
    if model_type == "RandomForest":
        print("Training RandomForest with Grid Search...")
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [11, 22],
            'max_depth': [2, 3],
            'min_samples_split': [20, 30],
            'min_samples_leaf': [15, 20]
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train_noisy)
        model = grid_search.best_estimator_
        
        y_val_pred = model.predict(X_val_scaled)
        val_r2 = r2_score(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_mape = np.mean(np.abs((y_val - y_val_pred) / (y_val + 1e-10))) * 100

        y_test_pred = model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mape = np.mean(np.abs((y_test - y_test_pred) / (y_test + 1e-10))) * 100

        rf_cv = grid_search.best_score_
        log_experiment(model_type, grid_search.best_params_, val_r2, test_r2, test_rmse, test_mae, test_mape)

        coefficients = model.feature_importances_
        shap_values = None
        if use_shap:
            print("Computing SHAP values...")
            shap_start = time.time()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_scaled)
            print(f"SHAP computation completed in {time.time() - shap_start:.2f} seconds")

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train_noisy)
        ridge_r2 = r2_score(y_test, ridge.predict(X_test_scaled))

    elif model_type == "Ridge":
        print("Training Ridge with Grid Search...")
        model = Ridge()
        param_grid = {
            'alpha': [150.0, 1500.0, 6000.0]
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train_noisy)
        model = grid_search.best_estimator_
        
        y_val_pred = model.predict(X_val_scaled)
        val_r2 = r2_score(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_mape = np.mean(np.abs((y_val - y_val_pred) / (y_val + 1e-10))) * 100

        y_test_pred = model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mape = np.mean(np.abs((y_test - y_test_pred) / (y_test + 1e-10))) * 100

        rf_cv = grid_search.best_score_
        log_experiment(model_type, grid_search.best_params_, val_r2, test_r2, test_rmse, test_mae, test_mape)

        coefficients = model.coef_
        shap_values = None
        ridge_r2 = None

    elif model_type == "XGBoost":
        print("Training XGBoost with Grid Search...")
        model = XGBRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [15, 30],
            'max_depth': [4, 6],
            'learning_rate': [0.01, 0.05],
            'reg_lambda': [5.0, 10.0]
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train_noisy)
        model = grid_search.best_estimator_
        
        y_val_pred = model.predict(X_val_scaled)
        val_r2 = r2_score(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_mape = np.mean(np.abs((y_val - y_val_pred) / (y_val + 1e-10))) * 100

        y_test_pred = model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mape = np.mean(np.abs((y_test - y_test_pred) / (y_test + 1e-10))) * 100

        rf_cv = grid_search.best_score_
        log_experiment(model_type, grid_search.best_params_, val_r2, test_r2, test_rmse, test_mae, test_mape)

        coefficients = model.feature_importances_
        shap_values = None
        if use_shap:
            print("Computing SHAP values...")
            shap_start = time.time()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_scaled)
            print(f"SHAP computation completed in {time.time() - shap_start:.2f} seconds")

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train_noisy)
        ridge_r2 = r2_score(y_test, ridge.predict(X_test_scaled))

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    training_time = time.time() - start_time
    print(f"Training {model_type} completed in {training_time:.2f} seconds")

    return (model, scaler, poly, selector, selected_features, model_type, val_r2, val_rmse, val_mae, val_mape,
            test_r2, test_rmse, test_mae, test_mape, coefficients, X_test_scaled, y_test, y_test_pred, shap_values, ridge_r2, rf_cv, training_time, total_train_size)

def preprocess_features(X, y, features):
    poly = PolynomialFeatures(degree=1, include_bias=False)
    X_poly = poly.fit_transform(X[features])
    
    selector = SelectKBest(score_func=mutual_info_regression, k=8)
    X_selected = selector.fit_transform(X_poly, y)
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    selected_indices = selector.get_support()
    selected_features = [f"poly_{i}" for i, selected in enumerate(selected_indices) if selected]
    
    return X_scaled, selected_features, poly, scaler, selector

def log_experiment(model_type, best_params, val_r2, test_r2, test_rmse, test_mae, test_mape):
    log_data = {
        'model_type': model_type,
        'best_params': str(best_params),
        'val_r2': val_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    log_df = pd.DataFrame([log_data])
    log_file = 'experiment_log.csv'
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)