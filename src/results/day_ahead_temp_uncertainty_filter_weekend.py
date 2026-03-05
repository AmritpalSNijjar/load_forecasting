import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
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

temp_uncertainties = [i for i in range (1, 11)]

model_names = ["linears", "xgb_best_per_hour", "xgb_best_max_hour", "lin_xgb_best_per_hour", "lin_xgb_best_max_hour"]

weekend_filter = lambda df: df["is_weekend"] == 1
weekday_filter = lambda df: df["is_weekend"] == 0

weekend_df = monte_carlo_temp_sensitivity(input_df_test, model_names, temp_uncertainties,
    n_simulations=100,
    base_t=60,
    filter_fn=weekend_filter
)

weekday_df = monte_carlo_temp_sensitivity(input_df_test, model_names, temp_uncertainties,
    n_simulations=100,
    base_t=60,
    filter_fn=weekday_filter
)

weekend_df["is_weekend"] = 1
weekday_df["is_weekend"] = 0

weekday_weekend_test_df = pd.concat([weekend_df, weekday_df], axis = 0)

save_loc = ROOT / "data" / "results" / "day_ahead_temp_uncertainty_filter_weekend.csv"

weekday_weekend_test_df.to_csv(save_loc)