import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import pickle

complete_train_loc = "../../data/processed/complete_train.csv"

complete_train_df = pd.read_csv(complete_train_loc)

X = complete_train_df.drop(columns = ["Load", "timestamp"])
y = complete_train_df["Load"]

complete_xgb = XGBRegressor(objective='reg:squarederror', n_estimators = 800, learning_rate=0.1, max_depth = 300, subsample=0.8)

complete_xgb.fit(X, y)

complete_xgb_model_loc = "../../trained_models/complete_xgb.pkl"

with open(complete_xgb_model_loc, 'wb') as file:
    pickle.dump(complete_xgb, file)