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

input_df_train = df_train[["timestamp", "Load", "Hour"] + ["temp_actual", "temp_6h_actual", "CDH_actual", "HDH_actual", "temp_actual_lag_24h", "Load_lag_24h", "Load_lag_48h", "is_weekend", "is_notable_day"]
]
input_df_test = df_test[["timestamp", "Load", "Hour"] + ["temp_actual", "temp_6h_actual", "CDH_actual", "HDH_actual", "temp_actual_lag_24h", "Load_lag_24h", "Load_lag_48h", "is_weekend", "is_notable_day"]
]

temp_uncertainties = [i for i in range (1, 6)]

model_names = ["linears", "xgb_best_per_hour", "xgb_best_max_hour", "lin_xgb_best_per_hour", "lin_xgb_best_max_hour"]

hour_filters = []

for hour in range(0, 24):
    hour_filter = lambda df: df["Hour"] == hour
    hour_filters.append(hour_filter)

print("\n\n", "*"*80, "*"*80)
print("Hour = 0")
print("*"*80, "*"*80, "\n\n")

hours_df_total = monte_carlo_temp_sensitivity(input_df_test, model_names, temp_uncertainties,
    n_simulations=100,
    base_t=60,
    filter_fn=hour_filters[0]
)

hours_df_total["Hour"] = 0

for hour in range(1, 24):

    print("\n\n", "*"*80, "*"*80)
    print(f"Hour = {hour}")
    print("*"*80, "*"*80, "\n\n")
    
    hour_df = monte_carlo_temp_sensitivity(input_df_test, model_names, temp_uncertainties, n_simulations=100, base_t=60, filter_fn=hour_filters[hour])

    hour_df["Hour"] = hour

    hours_df_total = pd.concat([hours_df_total, hour_df], axis = 0)


save_loc = ROOT / "data" / "results" / "day_ahead_temp_uncertainty_filter_hour.csv"

hours_df_total.to_csv(save_loc)