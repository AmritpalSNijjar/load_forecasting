from feature_utils import day_ahead_df_transformer, simulate_long_forecast
import pandas as pd

raw_train_loc = "../../data/raw/train.xlsx"
raw_test_loc = "../../data/raw/test.xlsx"

raw_train = pd.read_excel(raw_train_loc)
raw_test = pd.read_excel(raw_test_loc)

transformed_train = day_ahead_df_transformer(raw_train)
transformed_test = day_ahead_df_transformer(raw_test)

transformed_train.to_csv("../../data/processed/day_ahead_train.csv", index = False)
transformed_test.to_csv("../../data/processed/day_ahead_test.csv", index = False)