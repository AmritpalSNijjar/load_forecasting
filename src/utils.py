import pandas as pd
import numpy as np

import pickle

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor

from sklearn.metrics import root_mean_squared_error

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

def load_saved_models(model_name):

        match model_name:

            case "linears":
                with open(ROOT / "trained_models" / "day_ahead_linears.pkl", 'rb') as file:
                    day_ahead_linears = pickle.load(file)
                return [day_ahead_linears]
                
            case "xgb_best_per_hour":
                with open(ROOT / "trained_models" / "day_ahead_xgbs_best_per_hour_xgbs.pkl", 'rb') as file:
                    day_ahead_xgbs_best_per_hour_xgbs = pickle.load(file)
                return [day_ahead_xgbs_best_per_hour_xgbs]
            
            case "xgb_best_max_hour":
                with open(ROOT / "trained_models" / "day_ahead_xgbs_best_max_hour_xgbs.pkl", 'rb') as file:
                    day_ahead_xgbs_best_max_hour_xgbs = pickle.load(file)
                return [day_ahead_xgbs_best_max_hour_xgbs]
            
            case "lin_xgb_best_per_hour":
                day_ahead_linears = load_saved_models("linears")[0]

                with open(ROOT / "trained_models" / "day_ahead_lin_xgbs_best_per_hour_xgbs.pkl", 'rb') as file:
                    day_ahead_lin_xgbs_best_per_hour_xgbs = pickle.load(file)
                
                return [day_ahead_linears, day_ahead_lin_xgbs_best_per_hour_xgbs]

            case "lin_xgb_best_max_hour": 
                day_ahead_linears = load_saved_models("linears")[0]

                with open(ROOT / "trained_models" / "day_ahead_lin_xgbs_best_max_hour_xgbs.pkl", 'rb') as file:
                    day_ahead_lin_xgbs_best_max_hour_xgbs = pickle.load(file)

                return [day_ahead_linears, day_ahead_lin_xgbs_best_max_hour_xgbs]


def day_ahead_cv_rmse(model_name, input_df, splits_df_loc):

    splits_df = pd.read_csv(splits_df_loc)

    match model_name:
        case "linears":
            models = [LinearRegression() for hour in range(0, 24)]
            
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

def day_ahead_generate_all_predictions(model_name, input_df):

    is_lin_xgb = False
    
    if model_name == "lin_xgb_best_per_hour":
        is_lin_xgb = True
    elif model_name == "lin_xgb_best_max_hour":
        is_lin_xgb = True

    preds_list = []

    models = load_saved_models(model_name)

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


temp_uncertainties = [i for i in range (1, 11)]

model_names = ["linears", "ridges", "lassos", "xgb_best_per_hour", "xgb_best_max_hour", "lin_xgb_best_per_hour", "lin_xgb_best_max_hour"]

def monte_carlo_temp_sensitivity( input_df_test, model_names, temp_uncertainties, n_simulations=100, base_t=60, filter_fn=None):
    
    temp_uncertainties_errors_dict = {
        "model_name": [],
        "temp_uncertainty": [],
        "test_error": [],
        "test_std": []
    }
    
    input_df_test["timestamp"] = pd.to_datetime(input_df_test["timestamp"])


    test_df_component_1 = input_df_test[["timestamp", "Load", "Hour"]]
    test_df_component_2 = input_df_test[["temp_actual_lag_24h", "Load_lag_24h", "Load_lag_48h", "is_weekend", "is_notable_day"]]

    print("=" * 80)
    print("MONTE CARLO SENSITIVITY ANALYSIS: LOAD FORECAST VS TEMPERATURE UNCERTAINTY")
    print(f"Total Models: {len(model_names)} | Uncertainty Levels: {len(temp_uncertainties)} | Simulations per Level: {n_simulations}")
    print("=" * 80)

    for model_idx, model_name in enumerate(model_names, 1):

        # Model performance on test set with actual temperatures

        if filter_fn is not None:
            mask = filter_fn(input_df_test)
            df_eval = input_df_test.loc[mask]
        else:
            df_eval = input_df_test

        if len(df_eval) == 0:
            continue

        preds = day_ahead_generate_all_predictions(model_name, df_eval)
        test_error_actual = root_mean_squared_error(preds, df_eval["Load"])

        temp_uncertainties_errors_dict["model_name"].append(model_name)
        temp_uncertainties_errors_dict["temp_uncertainty"].append(0)
        temp_uncertainties_errors_dict["test_error"].append(test_error_actual)
        temp_uncertainties_errors_dict["test_std"].append(0.0)

        print("\n" + "-" * 80)
        print(f"[MODEL {model_idx}/{len(model_names)}] Evaluating: {model_name}")
        print("-" * 80)

        for unc_idx, temp_uncertainty in enumerate(temp_uncertainties, 1):

            print(
                f"\n  → Temperature Uncertainty σ = {temp_uncertainty}°F "
                f"({unc_idx}/{len(temp_uncertainties)}) | Running {n_simulations} Monte Carlo sims..."
            )

            temp_uncertainties_errors_dict["model_name"].append(model_name)
            temp_uncertainties_errors_dict["temp_uncertainty"].append(temp_uncertainty)

            rmse_list = []

            for i in range(n_simulations):
                simulated_temp = simulate_long_forecast(
                    input_df_test["temp_actual"],
                    std=temp_uncertainty,
                    correlation_factor=0.5
                )

                test_df_simulated = pd.DataFrame(simulated_temp)
                test_df_simulated["temp_6h_simulated"] = test_df_simulated["temp_simulated"].rolling(6).mean()
                test_df_simulated = test_df_simulated.dropna()

                test_df_simulated["CDH_simulated"] = (test_df_simulated["temp_simulated"] - base_t).clip(lower=0)
                test_df_simulated["HDH_simulated"] = (base_t - test_df_simulated["temp_simulated"]).clip(lower=0)

                valid_index = test_df_simulated.index

                input_df_simulated = pd.concat(
                    [
                        test_df_component_1.loc[valid_index],
                        test_df_simulated,
                        test_df_component_2.loc[valid_index],
                    ],
                    axis=1,
                )

                input_df_simulated = input_df_simulated.rename(
                    columns={
                        "temp_simulated": "temp_actual",
                        "temp_6h_simulated": "temp_6h_actual",
                        "CDH_simulated": "CDH_actual",
                        "HDH_simulated": "HDH_actual",
                    }
                )

                if filter_fn is not None:
                    mask = filter_fn(input_df_simulated)
                    df_eval = input_df_simulated.loc[mask]
                else:
                    df_eval = input_df_simulated

                if len(df_eval) == 0:
                    continue

                preds = day_ahead_generate_all_predictions(model_name, df_eval)
                rmse = root_mean_squared_error(preds, df_eval["Load"])
                rmse_list.append(rmse)

            temp_uncertainties_errors_dict["test_error"].append(np.mean(rmse_list))
            temp_uncertainties_errors_dict["test_std"].append(np.std(rmse_list))

            print(
                f"  ✔ Completed σ = {temp_uncertainty}°F | "
                f"Mean RMSE: {np.mean(rmse_list):.4f} | Std: {np.std(rmse_list):.4f}"
            )

    return pd.DataFrame(temp_uncertainties_errors_dict)

