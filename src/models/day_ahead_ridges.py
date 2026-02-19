import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import pickle

day_ahead_train_loc = "../../data/processed/day_ahead_train.csv"

day_ahead_train_df = pd.read_csv(day_ahead_train_loc)

features_to_train = ["temp_actual", "temp_6h_actual", "CDH_actual", "HDH_actual", "temp_actual_lag_24h", "Load_lag_24h", "Load_lag_48h", "is_weekend", "is_notable_day"]

# see ../../day_ahead_modelling.ipynb
best_alpha = 32 

# ridge_reg_per_hour

ridge_regs = []

for hour in range(0, 24):
    hourly_lin_model = Ridge(alpha = best_alpha)

    hour_df = day_ahead_train_df[day_ahead_train_df["Hour"] == hour]

    X_hour = hour_df[features_to_train]
    y_hour = hour_df["Load"]

    hourly_lin_model.fit(X_hour, y_hour)

    ridge_regs.append(hourly_lin_model)

# save

ridge_regressions_loc = "../../trained_models/day_ahead_ridges.pkl"

with open(ridge_regressions_loc, 'wb') as file:
    pickle.dump(ridge_regs, file)