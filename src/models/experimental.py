import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

baseline_train_loc = "../../data/processed/baseline_train.csv"
splits_df_loc = "../../data/splits/split_bounds.csv"

baseline_train_df = pd.read_csv(baseline_train_loc, parse_dates=["timestamp"])
splits_df = pd.read_csv(splits_df_loc)

rmse_list = []

for i in range(1, 6):
    split = f"split_{i}"

    row = splits_df[splits_df["split"] == split].iloc[0]
    train_start_date = row["train_start_date"]
    train_end_date = row["train_end_date"]
    val_start_date = row["val_start_date"]
    val_end_date = row["val_end_date"]

    train_mask = (baseline_train_df["timestamp"] >= train_start_date) & (baseline_train_df["timestamp"] < train_end_date)
    val_mask = (baseline_train_df["timestamp"] >= val_start_date) & (baseline_train_df["timestamp"] < val_end_date)
    
    train_split = baseline_train_df[train_mask]
    val_split = baseline_train_df[val_mask]

    X_train = train_split[["Month_sin", "Month_cos", "Day", "Hour_sin", "Hour_cos", "avg_region_temp", "avg_region_ghi"]]
    y_train = train_split["Load"]

    X_val = val_split[["Month_sin", "Month_cos", "Day", "Hour_sin", "Hour_cos", "avg_region_temp", "avg_region_ghi"]]
    y_val = val_split["Load"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)

    rmse_list.append(rmse)
    
print(f"Average RMSE across splits: {np.mean(rmse_list):.2f} ± {np.std(rmse_list):.2f}")