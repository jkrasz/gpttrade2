import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries
from config import API_KEY, DATA_FILE
from sklearn.preprocessing import MinMaxScaler
import ta
import pytz
import os
from config import symbol, DATA_FILE, CONDITIONS_FILE

# Improved market open check function
def is_market_open():
    ny_time = datetime.now(pytz.timezone('America/New_York'))
    opening_time = pytz.timezone('America/New_York').localize(datetime(ny_time.year, ny_time.month, ny_time.day, 9, 30))
    closing_time = pytz.timezone('America/New_York').localize(datetime(ny_time.year, ny_time.month, ny_time.day, 16, 0))
    return opening_time <= ny_time <= closing_time and ny_time.weekday() < 5

def fetch_current_price(symbol):
    ts = TimeSeries(API_KEY, output_format='pandas')
    data, _ = ts.get_quote_endpoint(symbol)
    if '05. price' in data.columns:
        return float(data['05. price'].iloc[0])
    raise ValueError("Current price not found in response.")

def fetch_data(symbol='GPRO', lookback_period=365):
    ts = TimeSeries(API_KEY, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='full')
    data.index = pd.to_datetime(data.index)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_period)
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    if data.empty:
        raise ValueError("No data fetched for the symbol with the specified lookback period.")
    data.replace(0, np.nan, inplace=True)
    data.dropna(inplace=True)
    column_mapping = {'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'}
    data.rename(columns=column_mapping, inplace=True)
    data = ta.add_all_ta_features(data, "open", "high", "low", "close", "volume", fillna=True)
    return data


def preprocess_data(data, sequence_length=60):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data needs to be a pandas DataFrame")
    
    # Assume the data includes the 'close' prices along with other features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['close']])  # Simplified for clarity; adapt as needed

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler

def save_data(predicted, actual):
    """Append the predicted and actual prices to a CSV file for persistence."""
    df = pd.DataFrame({'Date': [datetime.now().strftime('%Y-%m-%d')], 'Predicted': [predicted], 'Actual': [actual]})
    if os.path.exists(DATA_FILE):
        df.to_csv(DATA_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(DATA_FILE, mode='w', header=True, index=False)

def load_data():
    """Load historical data from a CSV file."""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=['Date', 'Predicted', 'Actual'])