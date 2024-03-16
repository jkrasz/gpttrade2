# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime, timedelta, time
import pytz
from time import sleep
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import os
from dotenv import load_dotenv

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
               bollinger_band_std_dev=2, price_jump_threshold=1.03):

    # Extracting indicator values from current_data with defaults to avoid KeyError
    rsi = current_data.get('momentum_rsi', 100)  # Default to 100, which is not oversold
    macd = current_data.get('trend_macd', 0)
    macd_signal = current_data.get('trend_macd_signal', 0)
    volume = current_data.get('Volume', 0)
    close_price = current_data.get('Close', 0)
    bb_lower_band = current_data.get('volatility_bbl', np.inf)  # Default to high to avoid false low band signals

    # Standard conditions
    volume_condition = volume > average_volume * threshold_volume_increase
    ema_condition = ema_short > ema_long
    rsi_condition = rsi < threshold_rsi_buy
    macd_condition = (macd > macd_signal) and (macd > macd_signal_threshold)
    bollinger_condition = close_price <= bb_lower_band

    # AI-driven condition: Predicted future price indicates significant jump
    ai_condition = predicted_close_price > close_price * price_jump_threshold
    
    # Combining all conditions to form a final buy signal
    buy_signal = volume_condition and ema_condition and rsi_condition and macd_condition and bollinger_condition and ai_condition
    
    return buy_signal


# Sell conditions function
def should_sell(current_data, buy_price, stop_loss_percent=0.10, take_profit_percent=0.15, threshold_rsi_sell=70):
    current_price = current_data.get('Close', 0)
    rsi = current_data.get('momentum_rsi', 0)
    sell_signal = current_price <= buy_price * (1 - stop_loss_percent) or \
                  current_price >= buy_price * (1 + take_profit_percent) or \
                  rsi > threshold_rsi_sell
    return sell_signal

def fetch_data(symbol='GPRO', lookback_period=365):
    required_data_points = 20  # Adjust based on your indicator needs
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_period)
    data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

    if data.empty:
        raise ValueError("No data fetched for the symbol with the specified lookback period.")

    data.replace(0, np.nan, inplace=True)
    data.dropna(inplace=True)

    if len(data) < required_data_points:
        raise ValueError("Not enough data for technical analysis. Increase lookback period.")

    data = ta.add_all_ta_features(data, "Open", "High", "Low", "Close", "Volume", fillna=True)
    
    return data

# Build LSTM model
def build_lstm_model(input_shape, units=50, dropout_rate=0.2):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
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
        'Close', 'Volume', 'momentum_rsi', 'trend_macd', 'trend_macd_signal',
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
    load_dotenv() # Load environment variables
    symbol = 'GPRO'
    data = fetch_data(symbol)
    processed_data, scaler = preprocess_data(data)
    input_shape = (processed_data.shape[1], 1)  # features, 1 time step
    print("buil model")
    lstm_model = build_lstm_model(input_shape)
    X_train = processed_data[:-1].reshape(-1, processed_data.shape[1], 1)
    y_train = data['Close'].values[1:]
    print("fit model")
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)

    in_position = False
    buy_price = 0  # Initialize buy_price to track the price at which we bought

    while True:
        print("In while loop")
        #if is_market_open(): 
        if True:  # Replace 'True' with actual market open check
            latest_data = fetch_data(symbol, lookback_period=60)  # Fetching the last 60 days
            if not latest_data.empty:
                latest_processed, _ = preprocess_data(latest_data)
                latest_scaled = latest_processed[-1].reshape(-1, latest_processed.shape[1], 1)
                predicted_close_price = lstm_model.predict(latest_scaled)[-1, 0]
                current_data = latest_data.iloc[-1].to_dict()
                predicted_close_price = lstm_model.predict(latest_scaled)[-1, 0]


                avg_volume = latest_data['Volume'].rolling(window=20).mean().iloc[-1]
                ema_short = latest_data['Close'].ewm(span=12, adjust=False).mean().iloc[-1]
                ema_long = latest_data['Close'].ewm(span=26, adjust=False).mean().iloc[-1]

                if not in_position and should_buy(predicted_close_price, current_data, avg_volume, ema_short, ema_long, threshold_rsi_buy=30, threshold_volume_increase=1.2, macd_signal_threshold=0, bollinger_band_window=20, bollinger_band_std_dev=2):
                    in_position = True
                    buy_price = current_data['Close']
                    print(f"Buy at {buy_price}")
                elif in_position and should_sell(current_data, buy_price, stop_loss_percent=0.10, take_profit_percent=0.15, threshold_rsi_sell=70):
                    in_position = False
                    sell_price = current_data['Close']
                    print(f"Sell at {sell_price}")
                    buy_price = 0  # Reset buy_price after selling
            else:
                print("Latest data is empty. Skipping this cycle.")

            print("Wait a 10 minuts before the next iteration")
            sleep(600)  # Wait a minute before the next iteration
        else:
            print("Market closed. Waiting...")
            sleep(300)  # Check every 5 minutes

if __name__ == "__main__":
    main()
