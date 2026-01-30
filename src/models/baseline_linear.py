import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

baseline_train_loc = "../../data/processed/baseline_train.csv"

baseline_train_df = pd.read_csv(baseline_train_loc)

X = baseline_train_df[["Month_sin", "Month_cos", "Day", "Hour_sin", "Hour_cos", "avg_region_temp", "avg_region_ghi"]]
y = baseline_train_df["Load"]

baseline_linear = LinearRegression()

baseline_linear.fit(X, y)

baseline_model_loc = "../../trained_models/baseline_linear.pkl"

with open(baseline_model_loc, 'wb') as file:
    pickle.dump(baseline_linear, file)