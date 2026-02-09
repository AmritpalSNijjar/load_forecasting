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

n_train_samples = len(X)
initial_len = int(0.6 * n_train_samples)
step_size = 24

oos_residuals = np.full(n_train_samples, np.nan)

for start in range(initial_len, n_train_samples, step_size):
    end = min(start + step_size, n_train_samples)

    wf_lin = LinearRegression()
    wf_lin.fit(X.iloc[:start], y.iloc[:start])

    preds = wf_lin.predict(X.iloc[start:end])

    oos_residuals[start:end] = y.iloc[start:end] - preds

mask = ~np.isnan(oos_residuals)

xgboost_component = XGBRegressor(objective='reg:squarederror', n_estimators = 3, learning_rate = 0.05, max_depth = 50, subsample = 0.8, random_state=12)
xgboost_component.fit(X.iloc[mask], oos_residuals[mask])

linear_component = LinearRegression()
linear_component.fit(X[features_to_train_lin], y)

lin_xgb_linear_loc = "../../trained_models/lin_xgb_linear_component.pkl"
lin_xgb_xgb_loc = "../../trained_models/lin_xgb_xgboost_component.pkl"

with open(lin_xgb_linear_loc, 'wb') as file:
    pickle.dump(linear_component, file)

with open(lin_xgb_xgb_loc, 'wb') as file:
    pickle.dump(xgboost_component, file)