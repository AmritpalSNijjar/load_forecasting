import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import pickle

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


day_ahead_train_loc = ROOT / "data" / "processed" / "day_ahead_train.csv"

day_ahead_train_df = pd.read_csv(day_ahead_train_loc)

features_to_train = ["temp_actual", "temp_6h_actual", "CDH_actual", "HDH_actual", "temp_actual_lag_24h", "Load_lag_24h", "Load_lag_48h", "is_weekend", "is_notable_day"]

# best_per_hour 

best_per_hour_hyperparams = pd.read_csv(ROOT / "src" / "models" / "best_per_hour_xgb_hyperparams.csv")

best_per_hour_xgbs = []

for hour in range(0, 24):
    hour_df = day_ahead_train_df[day_ahead_train_df["Hour"] == hour]

    X_hour = hour_df[features_to_train]
    y_hour = hour_df["Load"]

    n_estimators = best_per_hour_hyperparams.iloc[hour]["n_estimators"]
    max_depth = best_per_hour_hyperparams.iloc[hour]["max_depth"]
    min_child_weight = best_per_hour_hyperparams.iloc[hour]["min_child_weight"]
    learning_rate = best_per_hour_hyperparams.iloc[hour]["learning_rate"]


    hour_xgb = XGBRegressor(objective='reg:squarederror', n_estimators = int(n_estimators), learning_rate=learning_rate, max_depth = int(max_depth), min_child_weight = int(min_child_weight), subsample=0.8, random_state = 12)

    hour_xgb.fit(X_hour, y_hour)

    best_per_hour_xgbs.append(hour_xgb)

# best_max_hour 

best_max_hour_hyperparams = pd.read_csv(ROOT / "src" / "models" / "best_max_hour_xgb_hyperparams.csv")

best_max_hour_xgbs = []

n_estimators = best_max_hour_hyperparams.iloc[0]["n_estimators"]
max_depth = best_max_hour_hyperparams.iloc[0]["max_depth"]
min_child_weight = best_max_hour_hyperparams.iloc[0]["min_child_weight"]
learning_rate = best_max_hour_hyperparams.iloc[0]["learning_rate"]

for hour in range(0, 24):
    hour_df = day_ahead_train_df[day_ahead_train_df["Hour"] == hour]

    X_hour = hour_df[features_to_train]
    y_hour = hour_df["Load"]

    hour_xgb = XGBRegressor(objective='reg:squarederror', n_estimators = int(n_estimators), learning_rate=learning_rate, max_depth = int(max_depth), min_child_weight = int(min_child_weight), subsample=0.8, random_state = 12)

    hour_xgb.fit(X_hour, y_hour)

    best_max_hour_xgbs.append(hour_xgb)

# save

best_per_hour_model_loc = ROOT / "trained_models" / "day_ahead_xgbs_best_per_hour_xgbs.pkl"
best_max_hour_model_loc = ROOT / "trained_models" / "day_ahead_xgbs_best_max_hour_xgbs.pkl"

with open(best_per_hour_model_loc, 'wb') as file:
    pickle.dump(best_per_hour_xgbs, file)

with open(best_max_hour_model_loc , 'wb') as file:
    pickle.dump(best_max_hour_xgbs, file)