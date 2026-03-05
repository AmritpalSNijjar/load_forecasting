from src.features.feature_utils import day_ahead_df_transformer
import pandas as pd

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

raw_train_loc = ROOT / "data" / "raw" / "Train.xlsx"
raw_test_loc = ROOT / "data" / "raw" / "Test.xlsx"

raw_train = pd.read_excel(raw_train_loc)
raw_test = pd.read_excel(raw_test_loc)

transformed_train = day_ahead_df_transformer(raw_train)
transformed_test = day_ahead_df_transformer(raw_test)

transformed_train.to_csv(ROOT / "data" / "processed" / "day_ahead_train.csv", index=False)
transformed_test.to_csv(ROOT / "data" / "processed" / "day_ahead_test.csv", index=False)