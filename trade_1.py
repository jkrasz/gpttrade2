# Import necessary libraries
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta, time
import pytz
from time import sleep
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import os
from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries
from tensorflow.keras.layers import Bidirectional
from datetime import date

def fetch_current_price(symbol='GPRO'):
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        raise ValueError("Alpha Vantage API key not found.")

    ts = TimeSeries(api_key, output_format='pandas')
    data, _ = ts.get_quote_endpoint(symbol)
    
    # Check the new structure - it seems '05. price' is a column, not an index
    print("Processed data structure:", data)

    # Data is now expected to be in the columns, not the index
    if '05. price' in data.columns:
        # The actual price is under the 'Global Quote' column, adjust accordingly
        current_price = float(data['05. price']['Global Quote'])
    else:
        raise ValueError(f"Current price not found in response: {data}")

    return current_price


def fetch_data(symbol='GPRO', lookback_period=365):
    load_dotenv()  # Load environment variables
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        raise ValueError("Alpha Vantage API key not found.")

    ts = TimeSeries(api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
    
    # Convert the index to datetime to ensure compatibility with your existing code
    data.index = pd.to_datetime(data.index)

    # Filter the data according to the lookback period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_period)
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    print(data)
    if data.empty:
        raise ValueError("No data fetched for the symbol with the specified lookback period.")

    data.replace(0, np.nan, inplace=True)  # Replace 0s with NaN to avoid misleading calculations
    data.dropna(inplace=True)  # Drop rows with NaN values

    # Convert column names to lowercase
    # Explicitly map old column names to new ones
    column_mapping = {
        '1. open': 'open', 
        '2. high': 'high', 
        '3. low': 'low', 
        '4. close': 'close', 
        '5. volume': 'volume'
    }
    data.rename(columns=column_mapping, inplace=True)


    if len(data) < 20:
        raise ValueError("Not enough data for technical analysis. Increase lookback period.")

    # Add technical analysis features with TA-Lib
    print(data.columns)  # This should print all column names in lowercase
    data = ta.add_all_ta_features(data, "open", "high", "low", "close", "volume", fillna=True)

    return data


# Improved market open check function
def is_market_open():
    ny_time = datetime.now(pytz.timezone('America/New_York'))
    opening_time, closing_time = pytz.timezone('America/New_York').localize(datetime(ny_time.year, ny_time.month, ny_time.day, 9, 30)), pytz.timezone('America/New_York').localize(datetime(ny_time.year, ny_time.month, ny_time.day, 16, 0))
    return opening_time <= ny_time <= closing_time and ny_time.weekday() < 5

# Buy conditions function
# Enhanced Buy conditions function
def should_buy(predicted_close_price, current_data, average_volume, ema_short, ema_long, 
               threshold_rsi_buy=30, threshold_volume_increase=1.2, 
               macd_signal_threshold=0, bollinger_band_window=20, 
               bollinger_band_std_dev=2, current_price=None, price_jump_threshold=1.03):  # Set a default value


    # Default to avoid error if not found or not provided
    rsi = current_data.get('momentum_rsi', 100)
    macd = current_data.get('trend_macd', 0)
    macd_signal = current_data.get('trend_macd_signal', 0)
    volume = current_data.get('volume', 0)
    # Use provided current price if available, otherwise fall back to the last known close price
    close_price = current_price if current_price is not None else current_data.get('close', 0)
    bb_lower_band = current_data.get('volatility_bbl', np.inf)

    volume_condition = volume > average_volume * threshold_volume_increase
    ema_condition = ema_short > ema_long
    rsi_condition = rsi < threshold_rsi_buy
    macd_condition = (macd > macd_signal) and (macd > macd_signal_threshold)
    bollinger_condition = close_price <= bb_lower_band
    ai_condition = predicted_close_price > close_price * price_jump_threshold
    
    buy_signal = volume_condition and ema_condition and rsi_condition and macd_condition and bollinger_condition and ai_condition
    return buy_signal


# Sell conditions function
def should_sell(current_data, buy_price, stop_loss_percent=0.10, take_profit_percent=0.15, threshold_rsi_sell=70):
    current_price = current_data.get('close', 0)
    rsi = current_data.get('momentum_rsi', 0)
    sell_signal = current_price <= buy_price * (1 - stop_loss_percent) or \
                  current_price >= buy_price * (1 + take_profit_percent) or \
                  rsi > threshold_rsi_sell
    return sell_signal

# Build LSTM model

def build_lstm_model(input_shape, units=50, dropout_rate=0.2):
    model = Sequential([
        Bidirectional(LSTM(units, return_sequences=True, input_shape=input_shape)),
        Dropout(dropout_rate),
        LSTM(units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units),
        Dropout(dropout_rate),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Preprocess data for LSTM and strategy use
def preprocess_data(data):
    scaler = StandardScaler()
    features = [
        'close', 'volume', 'momentum_rsi', 'trend_macd', 'trend_macd_signal',
        'volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbhi', 'volatility_bbli', 
        'volatility_kcc', 'volatility_kch', 'volatility_kcl', 'volatility_dcl', 'volatility_dch',
        'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow', 'volatility_atr', 'volume_mfi'
        ]
    # Check if all required features are present
    missing_cols = [col for col in features if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    scaled_data = scaler.fit_transform(data[features])
    return scaled_data, scaler
def main():
    load_dotenv()  # Load environment variables
    symbol = 'GPRO'
    
    last_data_fetch_date = None
    historical_data = None
    processed_data = None
    scaler = None
    lstm_model = None
    in_position = False
    buy_price = 0  # Initialize buy_price to track the price at which we bought

    while True:
        today = date.today()
        if last_data_fetch_date is None or last_data_fetch_date != today:
            print("Fetching new historical data...")
            historical_data = fetch_data(symbol)
            processed_data, scaler = preprocess_data(historical_data)
            input_shape = (processed_data.shape[1], 1)  # features, 1 time step
            lstm_model = build_lstm_model(input_shape)
            X_train = processed_data[:-1].reshape(-1, processed_data.shape[1], 1)
            y_train = historical_data['close'].values[1:]
            print("Fitting model with new data")
            lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)
            last_data_fetch_date = today

        print("In while loop")
        current_price = fetch_current_price(symbol)
        
        if is_market_open():
        #if True:
            if historical_data is not None and not historical_data.empty:
                latest_processed, _ = preprocess_data(historical_data)
                latest_scaled = latest_processed[-1].reshape(-1, latest_processed.shape[1], 1)
                predicted_close_price = lstm_model.predict(latest_scaled)[-1, 0]

                avg_volume = historical_data['volume'].rolling(window=20).mean().iloc[-1]
                ema_short = historical_data['close'].ewm(span=12, adjust=False).mean().iloc[-1]
                ema_long = historical_data['close'].ewm(span=26, adjust=False).mean().iloc[-1]

                # Use current price and historical data to decide whether to buy or sell
                if not in_position and should_buy(predicted_close_price, historical_data.iloc[-1].to_dict(), avg_volume, ema_short, ema_long, threshold_rsi_buy=30, threshold_volume_increase=1.2, macd_signal_threshold=0, bollinger_band_window=20, bollinger_band_std_dev=2, current_price=current_price):
                    in_position = True
                    buy_price = current_price
                    print(f"Buy at {buy_price}")
                elif in_position and should_sell(historical_data.iloc[-1].to_dict(), buy_price, stop_loss_percent=0.10, take_profit_percent=0.15, threshold_rsi_sell=70, current_price=current_price):
                    in_position = False
                    sell_price = current_price
                    print(f"Sell at {sell_price}")
                    buy_price = 0  # Reset buy_price after selling
            else:
                print("Historical data is empty. Skipping this cycle.")

            print("Wait for the next iteration")
            sleep(6000)  # Wait a defined time before the next iteration
        else:
            print("Market closed. Waiting...")
            sleep(3000)  # Check every 5 minutes

if __name__ == "__main__":
    main()