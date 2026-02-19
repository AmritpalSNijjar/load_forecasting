# Initial testing splits

import pandas as pd

# Define splits
splits = {
    "split_1": {
        "train_start_date": pd.Timestamp("2020-01-01 00:00:00"),
        "train_end_date":   pd.Timestamp("2020-10-01 00:00:00"),
        "val_start_date":   pd.Timestamp("2020-10-01 00:00:00"),
        "val_end_date":     pd.Timestamp("2021-01-01 00:00:00")
    },
    "split_2": {
        "train_start_date": pd.Timestamp("2020-01-01 00:00:00"),
        "train_end_date":   pd.Timestamp("2021-01-01 00:00:00"),
        "val_start_date":   pd.Timestamp("2021-01-01 00:00:00"),
        "val_end_date":     pd.Timestamp("2021-04-01 00:00:00")
    },
    "split_3": {
        "train_start_date": pd.Timestamp("2020-01-01 00:00:00"),
        "train_end_date":   pd.Timestamp("2021-04-01 00:00:00"),
        "val_start_date":   pd.Timestamp("2021-04-01 00:00:00"),
        "val_end_date":     pd.Timestamp("2021-07-01 00:00:00")
    },
    "split_4": {
        "train_start_date": pd.Timestamp("2020-01-01 00:00:00"),
        "train_end_date":   pd.Timestamp("2021-07-01 00:00:00"),
        "val_start_date":   pd.Timestamp("2021-07-01 00:00:00"),
        "val_end_date":     pd.Timestamp("2021-10-01 00:00:00")
    },
    "split_5": {
        "train_start_date": pd.Timestamp("2020-01-01 00:00:00"),
        "train_end_date":   pd.Timestamp("2021-10-01 00:00:00"),
        "val_start_date":   pd.Timestamp("2021-10-01 00:00:00"),
        "val_end_date":     pd.Timestamp("2022-01-01 00:00:00")
    }
}

# Convert to DataFrame
df_splits = pd.DataFrame.from_dict(splits, orient="index")
df_splits.index.name = "split"
df_splits.reset_index(inplace=True)

# Save as CSV
df_splits.to_csv("../data/splits/split_bounds.csv", index=False)