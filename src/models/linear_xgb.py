import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import pickle

complete_train_loc = "../../data/processed/complete_train.csv"

complete_train_df = pd.read_csv(complete_train_loc)

features_to_train_lin = ["Hour_sin", "Hour_cos", "Month", "Day", "temp_6h", "avg_region_ghi", "is_weekend", "is_notable_day", "CDH", "HDH", "Load_lag_1h", "Load_lag_2h", "Load_lag_3h", "Load_lag_24h"]

X = complete_train_df.drop(columns = ["Load", "timestamp"])
y = complete_train_df["Load"]

linear_component = LinearRegression()
linear_component.fit(X[features_to_train_lin], y)

residuals = y - linear_component.predict(X[features_to_train_lin])

xgboost_component = XGBRegressor( objective='reg:squarederror', n_estimators = 800, learning_rate = 0.05, max_depth = 4, subsample = 0.8)
xgboost_component.fit(X, residuals)

lin_xgb_linear_loc = "../../trained_models/lin_xgb_linear_component.pkl"
lin_xgb_xgb_loc = "../../trained_models/lin_xgb_xgboost_component.pkl"

with open(lin_xgb_linear_loc, 'wb') as file:
    pickle.dump(linear_component, file)

with open(lin_xgb_xgb_loc, 'wb') as file:
    pickle.dump(xgboost_component, file)