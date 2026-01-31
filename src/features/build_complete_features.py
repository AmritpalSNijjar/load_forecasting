from feature_utils import complete_df_transformer
import pandas as pd

raw_train_loc = "../../data/raw/train.xlsx"
raw_test_loc = "../../data/raw/test.xlsx"

raw_train = pd.read_excel(raw_train_loc)
raw_test = pd.read_excel(raw_test_loc)

transformed_train = complete_df_transformer(raw_train)
transformed_test = complete_df_transformer(raw_test)

transformed_train.to_csv("../../data/processed/complete_train.csv", index = False)
transformed_test.to_csv("../../data/processed/complete_test.csv", index = False)