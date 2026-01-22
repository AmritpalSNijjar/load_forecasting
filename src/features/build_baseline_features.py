from feature_utils import baseline_df_transformer
import pandas as pd

raw_train_loc = "../../data/raw/train.xlsx"
raw_test_loc = "../../data/raw/test.xlsx"

raw_train = pd.read_excel(raw_train_loc)
raw_test = pd.read_excel(raw_test_loc)

transformed_train = baseline_df_transformer(raw_train)
transformed_test = baseline_df_transformer(raw_test)

save_train_loc = transformed_train.to_csv("../../data/processed/baseline_train.csv", index = False)
save_test_loc = transformed_test.to_csv("../../data/processed/baseline_test.csv", index = False)