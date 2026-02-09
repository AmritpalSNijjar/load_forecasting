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
                        "2022-01-01", "2022-07-04", "2022-11-24", "2022-11-25", "2022-12-24", "2022-12-25", "2022-12-31"]

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

def complete_df_transformer(raw_data_df):

    YEARS_DICT = {1:2020, 2:2021, 3:2022}

    TEMP_COLS = [f"Site-{i + 1} Temp" for i in range(5)]
    GHI_COLS = [f"Site-{i + 1} GHI" for i in range(5)]

    # Holidays considered: New_Years_Day, Independence_Day, Thanksgiving_Day, Day_After_Thanksgiving, Christmas_Eve, Christmas_Day, New_Years_Eve
    list_of_notable_days = ["2020-01-01", "2020-07-04", "2020-11-26", "2020-11-27", "2020-12-24", "2020-12-25", "2020-12-31", 
                        "2021-01-01", "2021-07-04", "2021-11-25", "2021-11-26", "2021-12-24", "2021-12-25", "2021-12-31",
                        "2022-01-01", "2022-07-04", "2022-11-24", "2022-11-25", "2022-12-24", "2022-12-25", "2022-12-31"]

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

    # Average of the temperature for the last 3, 6, and 24 hours.
    df_transformed["temp_3h"] = df_transformed["avg_region_temp"].rolling(3).mean()
    df_transformed["temp_6h"] = df_transformed["avg_region_temp"].rolling(6).mean()
    df_transformed["temp_24h"] = df_transformed["avg_region_temp"].rolling(24).mean()

    # Cooling & Heating Degrees
    df_transformed["CDH"] = (df_transformed["avg_region_temp"] - base_t).clip(lower=0)
    df_transformed["HDH"] = (base_t - df_transformed["avg_region_temp"]).clip(lower=0) 
    
    # Aggregate of the Cooling Degree Hours for the last 3, 6, and 24 hours.
    df_transformed["CDH_3h"] = df_transformed["CDH"].rolling(3).sum()
    df_transformed["CDH_6h"] = df_transformed["CDH"].rolling(6).sum()
    df_transformed["CDH_24h"] = df_transformed["CDH"].rolling(24).sum()
    
    # Aggregate of the Heating Degree Hours for the last 3, 6, and 24 hours.
    df_transformed["HDH_3h"] = df_transformed["HDH"].rolling(3).sum()
    df_transformed["HDH_6h"] = df_transformed["HDH"].rolling(6).sum()
    df_transformed["HDH_24h"] = df_transformed["HDH"].rolling(24).sum()

    # Ensure chronological ordering.
    df_transformed = df_transformed.sort_values("timestamp")

    df_transformed["Load_lag_1h"] = df_transformed["Load"].shift(1)
    df_transformed["Load_lag_2h"] = df_transformed["Load"].shift(2)
    df_transformed["Load_lag_3h"] = df_transformed["Load"].shift(3)
    df_transformed["Load_lag_6h"] = df_transformed["Load"].shift(6)
    df_transformed["Load_lag_12h"] = df_transformed["Load"].shift(12)
    df_transformed["Load_lag_24h"] = df_transformed["Load"].shift(24)
    df_transformed["Load_lag_48h"] = df_transformed["Load"].shift(48)
    df_transformed["Load_lag_168h"] = df_transformed["Load"].shift(168)

    df_transformed = df_transformed.dropna()

    # Add feature indicating weekend.
    df_transformed['is_weekend'] = (df_transformed['timestamp'].dt.weekday >= 5).astype(int)

    # Add feature indicating notable days.
    df_transformed['is_notable_day'] = (df_transformed['timestamp'].dt.normalize().isin(notable_days)).astype(int)

    return df_transformed

def simulate_long_forecast(actual_temp_series, base_std = 1.0, max_std = 5.0, correlation_factor = 0.5):
    
    simulated_forecast = []
    last_forecast = actual_temp_series.iloc[0]
    
    hours_in_day = 24
    
    for i, actual in enumerate(actual_temp_series):
        hour_of_day = i % hours_in_day
        
        # Scale uncertainty linearly within the day
        hour_std = base_std + (max_std - base_std) * (hour_of_day / (hours_in_day - 1))
        
        # Random error for this hour
        error = np.random.normal(loc = 0, scale = hour_std)
        
        # Forecast combines correlation with last forecast and random error
        new_forecast = last_forecast + correlation_factor * (actual - last_forecast) + error
        
        simulated_forecast.append(new_forecast)
        last_forecast = new_forecast
    
    return pd.Series(simulated_forecast, index = actual_temp_series.index)

def day_ahead_df_transformer(raw_data_df):

    YEARS_DICT = {1:2020, 2:2021, 3:2022}

    TEMP_COLS = [f"Site-{i + 1} Temp" for i in range(5)]

    # Holidays considered: New_Years_Day, Independence_Day, Thanksgiving_Day, Day_After_Thanksgiving, Christmas_Eve, Christmas_Day, New_Years_Eve
    list_of_notable_days = ["2020-01-01", "2020-07-04", "2020-11-26", "2020-11-27", "2020-12-24", "2020-12-25", "2020-12-31", 
                        "2021-01-01", "2021-07-04", "2021-11-25", "2021-11-26", "2021-12-24", "2021-12-25", "2021-12-31",
                        "2022-01-01", "2022-07-04", "2022-11-24", "2022-11-25", "2022-12-24", "2022-12-25", "2022-12-31"]

    notable_days = pd.to_datetime(list_of_notable_days)

    n_temp_forecasts = 10

    base_t = 60

    df_transformed = raw_data_df.copy()

    # Transform temperature readings to Farenheit.
    df_transformed[TEMP_COLS] = df_transformed[TEMP_COLS].apply(lambda x: x * 9/5 + 32)
    
    # Average the temperature measurements.
    df_transformed["temp_actual"] = df_transformed[TEMP_COLS].mean(axis=1)

    # Generate n_temp_forecasts=10 simulated temperature forecasts.
    for i in range(n_temp_forecasts):
        temp_forecast = simulate_long_forecast(df_transformed["temp_actual"]).rename(f"temp_forecast_{i + 1}")
        df_transformed = pd.concat([df_transformed, temp_forecast], axis = 1)

    # Create timestamps.
    df_transformed['Year'] = df_transformed['Year'].map(YEARS_DICT)
    df_transformed['Hour'] = df_transformed['Hour'] - 1
    df_transformed['timestamp'] = pd.to_datetime(df_transformed[['Year', 'Month', 'Day', 'Hour']])

    df_transformed["Hour_sin"] = np.sin(2*np.pi*df_transformed["Hour"]/24)
    df_transformed["Hour_cos"] = np.cos(2*np.pi*df_transformed["Hour"]/24)

    # Average of the temperature over the last 6 hours.
    df_transformed["temp_6h_actual"] = df_transformed["temp_actual"].rolling(6).mean()

    # Cooling & Heating Degrees.
    df_transformed["CDH_actual"] = (df_transformed["temp_actual"] - base_t).clip(lower=0)
    df_transformed["HDH_actual"] = (base_t - df_transformed["temp_actual"]).clip(lower=0) 
    
    # Average of the temperature over the last 6 hours and Cooling & Heating Degrees for all 10 simulated teperature forecasts.
    for i in range(n_temp_forecasts):
        df_transformed[f"temp_6h_forecast_{i + 1}"] = df_transformed[f"temp_forecast_{i + 1}"].rolling(6).mean()

        df_transformed[f"CDH_forecast_{i + 1}"] = (df_transformed[f"temp_forecast_{i + 1}"] - base_t).clip(lower=0)
        df_transformed[f"HDH_forecast_{i + 1}"] = (base_t - df_transformed[f"temp_forecast_{i + 1}"]).clip(lower=0) 


    # Ensure chronological ordering.
    df_transformed = df_transformed.sort_values("timestamp")

    # Load 24h, 48h lags. Temp 24h lags.
    df_transformed["Load_lag_24h"] = df_transformed["Load"].shift(24)
    df_transformed["Load_lag_48h"] = df_transformed["Load"].shift(48)

    df_transformed["temp_actual_lag_24h"] = df_transformed["temp_actual"].shift(24)

    df_transformed = df_transformed.dropna()

    # Add feature indicating weekend.
    df_transformed['is_weekend'] = (df_transformed['timestamp'].dt.weekday >= 5).astype(int)

    # Add feature indicating notable days.
    df_transformed['is_notable_day'] = (df_transformed['timestamp'].dt.normalize().isin(notable_days)).astype(int)

    return df_transformed