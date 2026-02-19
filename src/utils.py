import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.feature_utils import *
from src.utils import *

# TO DO: Complete this function. right now there is a model_name which is "xgb_best_per_hour" and the hyperparams are "best_per_hour_xgb",
# need to fix this naming inconsistency before this function can work properly.

def get_hyperparams(model_name):

    return pd.read_csv(ROOT / "src" / "models" / model_name + "_hyperparams.csv")


best_per_hour_lin_xgb_hyperparams = pd.read_csv(ROOT / "src" / "models" / "best_per_hour_lin_xgb_hyperparams.csv") 
best_max_hour_lin_xgb_hyperparams = pd.read_csv(ROOT / "src" / "models" / "best_max_hour_lin_xgb_hyperparams.csv")
best_per_hour_xgb_hyperparams = pd.read_csv(ROOT / "src" / "models" / "best_per_hour_xgb_hyperparams.csv")
best_max_hour_xgb_hyperparams = pd.read_csv(ROOT / "src" / "models" / "best_max_hour_xgb_hyperparams.csv")


# TO DO: Save the hyperparams somewhere, instead of hard-coding here.
def get_ridge_alpha():
    return 32

def get_lasso_alpha():
    return 3.6

def simulate_long_forecast(actual_temp_series, std = 1.0, correlation_factor = 0.5):
    
    simulated_forecast = []
    last_forecast = actual_temp_series.iloc[0]
    
    hours_in_day = 24
    
    for i, actual in enumerate(actual_temp_series):
        hour_of_day = i % hours_in_day
        
        # Random error for this hour
        error = np.random.normal(loc = 0, scale = std)
        
        # Forecast combines correlation with last forecast and random error
        new_forecast = last_forecast + correlation_factor * (actual - last_forecast) + error
        
        simulated_forecast.append(new_forecast)
        last_forecast = new_forecast
    
    return pd.Series(simulated_forecast, index = actual_temp_series.index, name = "temp_simulated")

def day_ahead_cv_rmse(model_name, input_df, splits_df_loc):

    splits_df = pd.read_csv(splits_df_loc)

    match model_name:
        case "linears":
            models = [LinearRegression() for hour in range(0, 24)]
            
        case "ridges":
            models = [Ridge(alpha = get_ridge_alpha()) for hour in range(0, 24)]
            
        case "lassos":
            models = [Lasso(alpha = get_lasso_alpha()) for hour in range(0, 24)]
            
        case "xgb_best_per_hour":
            models = [XGBRegressor(objective='reg:squarederror', 
                                   n_estimators = int(best_per_hour_xgb_hyperparams.iloc[hour]["n_estimators"]), 
                                   learning_rate = best_per_hour_xgb_hyperparams.iloc[hour]["learning_rate"], 
                                   max_depth = int(best_per_hour_xgb_hyperparams.iloc[hour]["max_depth"]), 
                                   min_child_weight = int(best_per_hour_xgb_hyperparams.iloc[hour]["min_child_weight"]), 
                                   subsample = 0.8, 
                                   random_state = 12) 
                      for hour in range(0,24)]
        
        case "xgb_best_max_hour":
            models = [XGBRegressor(objective='reg:squarederror', 
                                   n_estimators = int(best_max_hour_xgb_hyperparams.iloc[0]["n_estimators"]), 
                                   learning_rate = best_max_hour_xgb_hyperparams.iloc[0]["learning_rate"], 
                                   max_depth = int(best_max_hour_xgb_hyperparams.iloc[0]["max_depth"]), 
                                   min_child_weight = int(best_max_hour_xgb_hyperparams.iloc[0]["min_child_weight"]), 
                                   subsample = 0.8, 
                                   random_state = 12) 
                      for hour in range(0,24)]
        case "lin_xgb_best_per_hour":
            lin_models = [LinearRegression() for hour in range(0, 24)]
            
            xgb_models = [XGBRegressor(objective='reg:squarederror', 
                                   n_estimators = int(best_per_hour_lin_xgb_hyperparams.iloc[hour]["n_estimators"]), 
                                   learning_rate = best_per_hour_lin_xgb_hyperparams.iloc[hour]["learning_rate"], 
                                   max_depth = int(best_per_hour_lin_xgb_hyperparams.iloc[hour]["max_depth"]), 
                                   min_child_weight = int(best_per_hour_lin_xgb_hyperparams.iloc[hour]["min_child_weight"]), 
                                   subsample = 0.8, 
                                   random_state = 12) 
                      for hour in range(0,24)]
        case "lin_xgb_best_max_hour":
            lin_models = [LinearRegression() for hour in range(0, 24)]
            
            xgb_models = [XGBRegressor(objective='reg:squarederror', 
                                   n_estimators = int(best_max_hour_lin_xgb_hyperparams.iloc[0]["n_estimators"]), 
                                   learning_rate = best_max_hour_lin_xgb_hyperparams.iloc[0]["learning_rate"], 
                                   max_depth = int(best_max_hour_lin_xgb_hyperparams.iloc[0]["max_depth"]), 
                                   min_child_weight = int(best_max_hour_lin_xgb_hyperparams.iloc[0]["min_child_weight"]), 
                                   subsample = 0.8, 
                                   random_state = 12) 
                      for hour in range(0,24)] 
            
    split_rmses = []
    
    for i in range(1, splits_df.shape[0] + 1):

        split = f"split_{i}"
        row = splits_df[splits_df["split"] == split].iloc[0]

        all_squared_errors = []
            
        for hour in range(0, 24):
            
            train_start_date = row["train_start_date"]
            train_end_date = row["train_end_date"]
            val_start_date = row["val_start_date"]
            val_end_date = row["val_end_date"]
            
            train_mask = (input_df["timestamp"] >= train_start_date) & (input_df["timestamp"] < train_end_date) & (input_df["Hour"] == hour)
            val_mask = (input_df["timestamp"] >= val_start_date) & (input_df["timestamp"] < val_end_date) & (input_df["Hour"] == hour)
            
            X_train = input_df[train_mask].drop(columns=["timestamp", "Load"])
            y_train = input_df[train_mask]["Load"]
            
            X_val = input_df[val_mask].drop(columns=["timestamp", "Load"])
            y_val = input_df[val_mask]["Load"]
            
            if model_name[:7] == "lin_xgb":
                
                n_train_samples = len(X_train)
                initial_len = int(0.6 * n_train_samples)
                step_size = 7
                
                oos_residuals = np.full(n_train_samples, np.nan)
        
                for start in range(initial_len, n_train_samples, step_size):
                    end = min(start + step_size, n_train_samples)
        
                    wf_lin = LinearRegression()
                    wf_lin.fit(X_train.iloc[:start], y_train.iloc[:start])
                
                    preds = wf_lin.predict(X_train.iloc[start:end])
        
                    oos_residuals[start:end] = y_train.iloc[start:end] - preds
        
                mask = ~np.isnan(oos_residuals)
        
                xgb_models[hour].fit(X_train.iloc[mask], oos_residuals[mask])
        
                lin_models[hour].fit(X_train, y_train)
            
                y_lin_pred = lin_models[hour].predict(X_val)
                y_xgbr_pred = xgb_models[hour].predict(X_val)
    
                y_pred = y_lin_pred + y_xgbr_pred
                
            else:
                models[hour].fit(X_train, y_train)
                y_pred = models[hour].predict(X_val)
    
            all_squared_errors.extend((y_val - y_pred) ** 2)
        
        split_rmse = np.sqrt(np.mean(all_squared_errors))
        split_rmses.append(split_rmse)
        
    return np.mean(split_rmses), np.std(split_rmses)

def day_ahead_generate_all_predictions(model_name, models, input_df):

    is_lin_xgb = False
    
    if model_name == "lin_xgb_best_per_hour":
        is_lin_xgb = True
    elif model_name == "lin_xgb_best_max_hour":
        is_lin_xgb = True

    preds_list = []

    for hour in range(24):
        hour_mask = input_df["Hour"] == hour
        X_hour = input_df.loc[hour_mask].drop(columns=["timestamp", "Load", "Hour"])

        if len(X_hour) == 0:
            continue

        if is_lin_xgb:
            y_pred_hour = models[0][hour].predict(X_hour) + models[1][hour].predict(X_hour)
        else:
            y_pred_hour = models[0][hour].predict(X_hour)

        preds_hour_df = pd.DataFrame({
            "timestamp": input_df.loc[hour_mask, "timestamp"],
            "prediction": y_pred_hour
        }, index=input_df.loc[hour_mask].index)

        preds_list.append(preds_hour_df)

    all_preds_df = pd.concat(preds_list)
    all_preds_df = all_preds_df.sort_index()
    return all_preds_df["prediction"]

