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
    

def engineered_df_transformer(raw_data_df):

    YEARS_DICT = {1:2020, 2:2021, 3:2022}

    TEMP_COLS = [f"Site-{i + 1} Temp" for i in range(5)]
    GHI_COLS = [f"Site-{i + 1} GHI" for i in range(5)]

    # Holidays considered: New_Years_Day, Independence_Day, Thanksgiving_Day, Day_After_Thanksgiving, Christmas_Eve, Christmas_Day, New_Years_Eve
    list_of_notable_days = ["2020-01-01", "2020-07-04", "2020-11-26", "2020-11-27", "2020-12-24", "2020-12-25", "2020-12-31", 
                        "2021-01-01", "2021-07-04", "2021-11-25", "2021-11-26", "2021-12-24", "2021-12-25", "2021-12-31",
                        "2023-01-01", "2023-07-04", "2023-11-23", "2023-11-24", "2023-12-24", "2023-12-25", "2023-12-31"]

    notable_days = pd.to_datetime(list_of_notable_days)

    base_t = 60

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

    # 6 hour rolling temperature average.
    df_transformed["temp_6h"] = df_transformed["avg_region_temp"].rolling(6).mean()

    # Cooling & Heating Degrees
    df_transformed["CDH"] = (df_transformed["avg_region_temp"] - base_t).clip(lower=0)
    df_transformed["HDH"] = (base_t - df_transformed["avg_region_temp"]).clip(lower=0) 

    # Ensure chronological ordering.
    df_transformed = df_transformed.sort_values("timestamp")

    # Load Lags
    df_transformed["Load_lag_1h"] = df_transformed["Load"].shift(1)
    df_transformed["Load_lag_2h"] = df_transformed["Load"].shift(2)
    df_transformed["Load_lag_3h"] = df_transformed["Load"].shift(3)
    df_transformed["Load_lag_24h"] = df_transformed["Load"].shift(24)

    df_transformed = df_transformed.dropna()

    # Add feature indicating weekend.
    df_transformed['is_weekend'] = (df_transformed['timestamp'].dt.weekday >= 5).astype(int)

    # Add feature indicating notable days.
    df_transformed['is_notable_day'] = (df_transformed['timestamp'].dt.normalize().isin(notable_days)).astype(int)

    cols_to_keep = ["Load", "Hour_sin", "Hour_cos", "Month", "Day", "temp_6h", "avg_region_ghi", "is_weekend", "is_notable_day", "CDH", "HDH", "Load_lag_1h", "Load_lag_2h", "Load_lag_3h", "Load_lag_24h"]
    df_transformed = df_transformed[["timestamp"] + cols_to_keep]

    return df_transformed
