import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator

def fetch_historical_data(ticker, start_date, end_date):
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No real data found for {ticker} in the specified range. Generating dummy data.")
            return _generate_dummy_data(start_date, end_date)
 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]
            if f'Close_{ticker}' in data.columns: 
                data = data.rename(columns={f'Close_{ticker}': 'Close', f'Volume_{ticker}': 'Volume'})
            elif 'Close' in data.columns: 
                pass 
            else:
                print("Could not find 'Close' column after flattening. Generating dummy data.")
                return _generate_dummy_data(start_date, end_date)
        
        required_cols = ['Close', 'Volume'] 
        if not all(col in data.columns for col in required_cols):
            print(f"Required columns {required_cols} not found. Generating dummy data.")
            return _generate_dummy_data(start_date, end_date)

        return data[['Close', 'Volume']] 
      

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}. Generating dummy data.")
        return _generate_dummy_data(start_date, end_date)

def _generate_dummy_data(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(42)

    base_price = 100
    daily_price_changes = np.random.normal(0, 0.005, len(dates)).cumsum()
    prices = base_price * np.exp(daily_price_changes)

    base_volume = 1_000_000
    daily_volume_changes = np.random.normal(0, 0.1, len(dates))
    volumes = base_volume * (1 + daily_volume_changes).clip(min=0.5)
    
    dummy_df = pd.DataFrame({
        'Close': prices, 
        'Volume': volumes
    }, index=dates)
    print("Generated dummy financial data.")
    return dummy_df

def calculate_and_discretize_features(data_df, returns_bins, rsi_window=14, rsi_bins=None, volume_bins=None):
    features_df = pd.DataFrame(index=data_df.index)
    
    features_df['Returns'] = data_df['Close'].pct_change() 
    
    rsi_indicator = RSIIndicator(close=data_df['Close'], window=rsi_window, fillna=True) 
    features_df['RSI'] = rsi_indicator.rsi()
    
    features_df['Volume'] = data_df['Volume']
    
    features_df['Returns_Category'] = pd.cut(
        features_df['Returns'], 
        bins=returns_bins, 
        labels=['Down', 'Flat', 'Up'],
        right=True, 
        include_lowest=True,
        ordered=False
    )

    if rsi_bins is None:
        rsi_bins = [0, 30, 70, 100]
        rsi_labels = ['Oversold', 'Neutral', 'Overbought']
    else:
        rsi_labels = [f'RSI_Cat{i}' for i in range(len(rsi_bins) - 1)]

    features_df['RSI_Category'] = pd.cut(
        features_df['RSI'], 
        bins=rsi_bins, 
        labels=rsi_labels, 
        right=True, 
        include_lowest=True,
        ordered=False
    )
    
    if volume_bins is None:
        volume_bins = features_df['Volume'].quantile([0, 0.33, 0.66, 1.0]).tolist()
        volume_labels = ['Low', 'Medium', 'High']
    else:
        volume_labels = [f'Vol_Cat{i}' for i in range(len(volume_bins) - 1)]

    features_df['Volume_Category'] = pd.cut(
        features_df['Volume'], 
        bins=volume_bins, 
        labels=volume_labels, 
        right=True, 
        include_lowest=True,
        ordered=False
    )

    features_df['State'] = features_df.apply(
        lambda row: (row['Returns_Category'], row['RSI_Category'], row['Volume_Category']), axis=1
    )
    
    initial_rows_before_drop = len(features_df)
    processed_data = features_df.dropna(subset=['Returns', 'Returns_Category', 'RSI', 'RSI_Category', 'Volume', 'Volume_Category', 'State']).copy()
    
    if len(processed_data) < initial_rows_before_drop:
        print(f"Dropped {initial_rows_before_drop - len(processed_data)} rows due to NaN values after feature calculation.")

    processed_data['Returns'] = pd.to_numeric(processed_data['Returns'])
    
    print(f"Data processed. Number of valid data points (days): {len(processed_data)}")
    
    return processed_data[['Returns', 'State']]