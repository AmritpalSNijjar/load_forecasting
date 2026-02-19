import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import pickle

day_ahead_train_loc = "../../data/processed/day_ahead_train.csv"

day_ahead_train_df = pd.read_csv(day_ahead_train_loc)

features_to_train = ["temp_actual", "temp_6h_actual", "CDH_actual", "HDH_actual", "temp_actual_lag_24h", "Load_lag_24h", "Load_lag_48h", "is_weekend", "is_notable_day"]

# lin_reg_per_hour

lin_regs = []

for hour in range(0, 24):
    hourly_lin_model = LinearRegression()

    hour_df = day_ahead_train_df[day_ahead_train_df["Hour"] == hour]

    X_hour = hour_df[features_to_train]
    y_hour = hour_df["Load"]

    hourly_lin_model.fit(X_hour, y_hour)

    lin_regs.append(hourly_lin_model)

# walk forward oos residuals for xgboost

hourly_residuals = {}
hourly_features = {}

for hour in range(24):
    hour_df = day_ahead_train_df[day_ahead_train_df["Hour"] == hour]

    X_hour = hour_df[features_to_train]
    y_hour = hour_df["Load"]

    n_train_samples = len(X_hour)
    initial_len = int(0.6 * n_train_samples)
    step_size = 7

    oos_residuals = np.full(n_train_samples, np.nan)

    for start in range(initial_len, n_train_samples, step_size):
        end = min(start + step_size, n_train_samples)
        wf_lin = LinearRegression()
        wf_lin.fit(X_hour.iloc[:start], y_hour.iloc[:start])
        preds = wf_lin.predict(X_hour.iloc[start:end])
        oos_residuals[start:end] = y_hour.iloc[start:end] - preds

    mask = ~np.isnan(oos_residuals)

    hourly_residuals[hour] = (X_hour.iloc[mask], oos_residuals[mask])


# best_per_hour 

best_per_hour_hyperparams = pd.read_csv("best_per_hour_lin_xgb_hyperparams.csv")

best_per_hour_xgbs = []

for hour in range(0, 24):
    
    X_resid, y_resid = hourly_residuals[hour]

    n_estimators = best_per_hour_hyperparams.iloc[hour]["n_estimators"]
    max_depth = best_per_hour_hyperparams.iloc[hour]["max_depth"]
    min_child_weight = best_per_hour_hyperparams.iloc[hour]["min_child_weight"]
    learning_rate = best_per_hour_hyperparams.iloc[hour]["learning_rate"]

    hour_xgb = XGBRegressor(objective='reg:squarederror', n_estimators = int(n_estimators), learning_rate=learning_rate, max_depth = int(max_depth), min_child_weight = int(min_child_weight), subsample=0.8, random_state = 12)
    hour_xgb.fit(X_resid, y_resid)

    best_per_hour_xgbs.append(hour_xgb)

# best_max_hour 

best_max_hour_hyperparams = pd.read_csv("best_max_hour_lin_xgb_hyperparams.csv")

best_max_hour_xgbs = []

n_estimators = best_max_hour_hyperparams.iloc[0]["n_estimators"]
max_depth = best_max_hour_hyperparams.iloc[0]["max_depth"]
min_child_weight = best_max_hour_hyperparams.iloc[0]["min_child_weight"]
learning_rate = best_max_hour_hyperparams.iloc[0]["learning_rate"]

for hour in range(0, 24):
    
    X_resid, y_resid = hourly_residuals[hour]

    hour_xgb = XGBRegressor(objective='reg:squarederror', n_estimators = int(n_estimators), learning_rate=learning_rate, max_depth = int(max_depth), min_child_weight = int(min_child_weight), subsample=0.8, random_state = 12)
    hour_xgb.fit(X_resid, y_resid)

    best_max_hour_xgbs.append(hour_xgb)

# save

linear_regressions_loc = "../../trained_models/day_ahead_linears.pkl"
best_per_hour_model_loc = "../../trained_models/day_ahead_lin_xgbs_best_per_hour_xgbs.pkl"
best_max_hour_model_loc = "../../trained_models/day_ahead_lin_xgbs_best_max_hour_xgbs.pkl"

with open(linear_regressions_loc, 'wb') as file:
    pickle.dump(lin_regs, file)

with open(best_per_hour_model_loc, 'wb') as file:
    pickle.dump(best_per_hour_xgbs, file)

with open(best_max_hour_model_loc , 'wb') as file:
    pickle.dump(best_max_hour_xgbs, file)