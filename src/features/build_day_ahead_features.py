from feature_utils import day_ahead_df_transformer, simulate_long_forecast
import pandas as pd

raw_train_loc = "../../data/raw/train.xlsx"
raw_test_loc = "../../data/raw/test.xlsx"

raw_train = pd.read_excel(raw_train_loc)
raw_test = pd.read_excel(raw_test_loc)

complete_df = pd.concat([raw_train, raw_test], axis = 0)

raw_train = complete_df[(complete_df["Year"] < 3) | ((complete_df["Year"] == 3) & ((complete_df["Month"] < 10) | ((complete_df["Month"] == 10) & (complete_df["Day"] <= 16))))]
raw_test = complete_df[(complete_df["Year"] == 3) & (((complete_df["Month"] == 10) & (complete_df["Day"] >= 15)) | (complete_df["Month"] > 10))]

transformed_train = day_ahead_df_transformer(raw_train)
transformed_test = day_ahead_df_transformer(raw_test)

transformed_train.to_csv("../../data/processed/day_ahead_train.csv", index = False)
transformed_test.to_csv("../../data/processed/day_ahead_test.csv", index = False)