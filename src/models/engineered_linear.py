import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

engineered_train_loc = "../../data/processed/engineered_train.csv"

engineered_train_df = pd.read_csv(engineered_train_loc)

X = engineered_train_df[["Hour_sin", "Hour_cos", "Month", "Day", "temp_6h", "avg_region_ghi", "is_weekend", "is_notable_day", "CDH", "HDH", "Load_lag_1h", "Load_lag_2h", "Load_lag_3h", "Load_lag_24h"]]
y = engineered_train_df["Load"]

engineered_linear = LinearRegression()

engineered_linear.fit(X, y)

engineered_model_loc = "../../trained_models/engineered_linear.pkl"

with open(engineered_model_loc, 'wb') as file:
    pickle.dump(engineered_linear, file)