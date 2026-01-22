import pandas as pd
import numpy as np

def baseline_df_transformer(raw_data_df):

    YEARS_DICT = {1:2020, 2:2021, 3:2022}

    TEMP_COLS = [f"Site-{i + 1} Temp" for i in range(5)]
    GHI_COLS = [f"Site-{i + 1} GHI" for i in range(5)]

    df_transformed = raw_data_df.copy()

    # Transform temperature readings to Farenheit.
    df_transformed[TEMP_COLS] = df_transformed[TEMP_COLS].apply(lambda x: x * 9/5 + 32)
    
    # Average the temperature and GHI measurements.
    df_transformed["avg_region_temp"] = df_transformed[TEMP_COLS].mean(axis=1)
    df_transformed["avg_region_ghi"] = df_transformed[GHI_COLS].mean(axis=1)

    # Create timestamps.
    df_transformed['Year'] = df_transformed['Year'].map(YEARS_DICT)
    df_transformed['Hour'] = df_transformed['Hour'] - 1
    df_transformed['timestamp'] = pd.to_datetime(df_transformed[['Year', 'Month', 'Day', 'Hour']])

    df_transformed["Hour_sin"] = np.sin(2*np.pi*df_transformed["Hour"]/24)
    df_transformed["Hour_cos"] = np.cos(2*np.pi*df_transformed["Hour"]/24)
    
    df_transformed["Month_sin"] = np.sin(2*np.pi*df_transformed["Month"]/24)
    df_transformed["Month_cos"] = np.cos(2*np.pi*df_transformed["Month"]/24)

    # Ensure chronological ordering.
    df_transformed = df_transformed.sort_values("timestamp")

    # Add feature indicating weekend.
    df_transformed['is_weekend'] = (df_transformed['timestamp'].dt.weekday >= 5).astype(int)

    # Rearrange columns

    cols_to_keep = ["Load", "Month_sin", "Month_cos", "Day", "Hour_sin", "Hour_cos", "avg_region_temp", "avg_region_ghi"]
    df_transformed = df_transformed[["timestamp"] + cols_to_keep]

    return df_transformed
    


