import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from sklearn.metrics import root_mean_squared_error

import sys
from pathlib import Path

# Get project root (two levels up from src/results/)
ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.feature_utils import *
from src.utils import *

features_to_train = ["temp_actual", "temp_6h_actual", "CDH_actual", "HDH_actual", "temp_actual_lag_24h", "Load_lag_24h", "Load_lag_48h", "is_weekend", "is_notable_day"]

train_loc = ROOT / "data" / "processed" / "day_ahead_train.csv"
test_loc = ROOT / "data" / "processed" / "day_ahead_test.csv"

df_train = pd.read_csv(train_loc)
df_test = pd.read_csv(test_loc)

with open(ROOT / "trained_models" / "day_ahead_linears.pkl", 'rb') as file:
    day_ahead_linears = pickle.load(file)

with open(ROOT / "trained_models" / "day_ahead_xgbs_best_per_hour_xgbs.pkl", 'rb') as file:
    day_ahead_xgbs_best_per_hour_xgbs = pickle.load(file)

with open(ROOT / "trained_models" / "day_ahead_xgbs_best_max_hour_xgbs.pkl", 'rb') as file:
    day_ahead_xgbs_best_max_hour_xgbs = pickle.load(file)

with open(ROOT / "trained_models" / "day_ahead_lin_xgbs_best_per_hour_xgbs.pkl", 'rb') as file:
    day_ahead_lin_xgbs_best_per_hour_xgbs = pickle.load(file)

with open(ROOT / "trained_models" / "day_ahead_lin_xgbs_best_max_hour_xgbs.pkl", 'rb') as file:
    day_ahead_lin_xgbs_best_max_hour_xgbs = pickle.load(file)


best_per_hour_lin_xgb_hyperparams = pd.read_csv(ROOT / "src" / "models" / "best_per_hour_lin_xgb_hyperparams.csv")
best_max_hour_lin_xgb_hyperparams = pd.read_csv(ROOT / "src" / "models" / "best_max_hour_lin_xgb_hyperparams.csv")
best_per_hour_xgb_hyperparams = pd.read_csv(ROOT / "src" / "models" / "best_per_hour_xgb_hyperparams.csv")
best_max_hour_xgb_hyperparams = pd.read_csv(ROOT / "src" / "models" / "best_max_hour_xgb_hyperparams.csv")

splits_df_loc = ROOT / "data" / "splits" / "split_bounds.csv"
splits_df = pd.read_csv(splits_df_loc)


input_df_train = df_train[["timestamp", "Load", "Hour"] + ["temp_actual", "temp_6h_actual", "CDH_actual", "HDH_actual", "temp_actual_lag_24h", "Load_lag_24h", "Load_lag_48h", "is_weekend", "is_notable_day"]
]
input_df_test = df_test[["timestamp", "Load", "Hour"] + ["temp_actual", "temp_6h_actual", "CDH_actual", "HDH_actual", "temp_actual_lag_24h", "Load_lag_24h", "Load_lag_48h", "is_weekend", "is_notable_day"]
]

model_names = ["linears", "xgb_best_per_hour", "xgb_best_max_hour", "lin_xgb_best_per_hour", "lin_xgb_best_max_hour"]
models = {"linears": [day_ahead_linears], "xgb_best_per_hour": [day_ahead_xgbs_best_per_hour_xgbs], "xgb_best_max_hour": [day_ahead_xgbs_best_max_hour_xgbs], "lin_xgb_best_per_hour": [day_ahead_linears, day_ahead_lin_xgbs_best_per_hour_xgbs], "lin_xgb_best_max_hour": [day_ahead_linears, day_ahead_lin_xgbs_best_max_hour_xgbs]}

errors_dict = {"model_name": model_names, "train_error" : [], "validation_error": [], "validation_std": [], "test_error": [], "test_error_pct_max": []}

max_load_train = input_df_train["Load"].max()
max_load_test = input_df_test["Load"].max()

for model_name in errors_dict["model_name"]:

    train_error = root_mean_squared_error(day_ahead_generate_all_predictions(model_name, input_df_train), input_df_train["Load"])

    validation_error, validation_std = day_ahead_cv_rmse(model_name, input_df_train, splits_df_loc = splits_df_loc)
    
    test_errors = []
    
    test_error = root_mean_squared_error(day_ahead_generate_all_predictions(model_name, input_df_test), input_df_test["Load"])
    test_errors.append(test_error)
    
    errors_dict["train_error"].append(train_error)
    errors_dict["validation_error"].append(validation_error)
    errors_dict["validation_std"].append(validation_std)
    errors_dict["test_error"].append(np.mean(test_errors))
    errors_dict["test_error_pct_max"].append(100*np.mean(test_errors)/max_load_test)

errors_df = pd.DataFrame(errors_dict)

save_loc = ROOT / "data" / "results" / "day_ahead_errors.csv"

errors_df.to_csv(save_loc)