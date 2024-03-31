# Import necessary libraries
import ta
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import pytz
from time import sleep
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional
import os
import logging
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
from tensorflow.keras.layers import BatchNormalization, Activation, Attention
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional, BatchNormalization, Activation
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf

# Initialization
load_dotenv()
logging.basicConfig(filename='stock_predictions.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
data_file = 'historical_data.csv'
# Ensure the logging directory exists
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/stock_predictions.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set API key and symbol
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
symbol = 'GPRO'

# Ensure logging directory exists
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/stock_predictions.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_current_price(symbol='GPRO'):
  
    if not api_key:
        raise ValueError("Alpha Vantage API key not found.")
    ts = TimeSeries(api_key, output_format='pandas')
    data, _ = ts.get_quote_endpoint(symbol)
    if '05. price' in data.columns:
        current_price = float(data['05. price'])
    else:
        raise ValueError("Current price not found in response: {}".format(data))
    return current_price

def fetch_data(symbol='GPRO', lookback_period=365):
    if not api_key:
        raise ValueError("Alpha Vantage API key not found.")
    ts = TimeSeries(api_key, output_format='pandas')
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

# Improved market open check function
def is_market_open():
    ny_time = datetime.now(pytz.timezone('America/New_York'))
    opening_time = pytz.timezone('America/New_York').localize(datetime(ny_time.year, ny_time.month, ny_time.day, 9, 30))
    closing_time = pytz.timezone('America/New_York').localize(datetime(ny_time.year, ny_time.month, ny_time.day, 16, 0))
    return opening_time <= ny_time <= closing_time and ny_time.weekday() < 5

# Buy conditions function
# Enhanced Buy conditions function
def should_buy(predicted_close_price, current_data, average_volume, ema_short, ema_long,
               threshold_rsi_buy=30, threshold_volume_increase=1.2,
               macd_signal_threshold=0, bollinger_band_window=20,
               bollinger_band_std_dev=2, current_price=None, price_jump_threshold=1.03,
               risk_tolerance=0.05, profit_tolerance=0.10):

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
    risk_condition = predicted_close_price > close_price * (1 - risk_tolerance)
    profit_condition = predicted_close_price > close_price * (1 + profit_tolerance)

    buy_signal = (volume_condition and ema_condition and rsi_condition and
                  macd_condition and bollinger_condition and ai_condition and
                  risk_condition and profit_condition)
    return buy_signal

# Sell conditions function
def should_sell(current_data, buy_price, stop_loss_percent=0.10, take_profit_percent=0.15, threshold_rsi_sell=70, current_price=None):
    # Use provided current price if available, otherwise fall back to the last known close price
    current_price = current_price if current_price is not None else current_data.get('close', 0)
    rsi = current_data.get('momentum_rsi', 0)
    sell_signal = current_price <= buy_price * (1 - stop_loss_percent) or \
                  current_price >= buy_price * (1 + take_profit_percent) or \
                  rsi > threshold_rsi_sell
    return sell_signal

# Define a custom simplified self-attention layer for time series
class SimpleSelfAttention(tf.keras.layers.Layer):
    def __init__(self, return_sequences=True):
        super(SimpleSelfAttention, self).__init__()
        self.return_sequences = return_sequences
    
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], input_shape[-1]),
                                 initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[-1],),
                                 initializer='uniform', trainable=True)
        super(SimpleSelfAttention, self).build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        
        if self.return_sequences:
            return output
        
        return tf.reduce_sum(output, axis=1)

def build_lstm_model(input_shape, units=64, dropout_rate=0.2, attention_units=32, l1_reg=0.01, l2_reg=0.01):
    model = Sequential([
        Conv1D(filters=64, kernel_size=5, padding='same', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), input_shape=input_shape),
        Activation('relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(units, return_sequences=True, activation='tanh')),
        Dropout(dropout_rate),
        Bidirectional(LSTM(units, return_sequences=True, activation='elu')),  # Using ELU activation for second LSTM layer
        Dropout(dropout_rate),
        SimpleSelfAttention(return_sequences=False),  # Applying the custom self-attention mechanism
        Dense(units, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        Dropout(dropout_rate),
        Dense(1)  # Predicting the next closing price
    ])
    model.compile(optimizer=RMSprop(), loss='mean_squared_error')  # Using RMSprop optimizer
    return model

# Preprocess data for LSTM and strategy use
def preprocess_data(data, sequence_length=60):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data needs to be a pandas DataFrame")

    # Ensure all required columns are present
    required_features = ['close', 'volume', 'momentum_rsi', 'trend_macd', 'trend_macd_signal', 
                         'volatility_bbh', 'volatility_bbl', 'volatility_bbm', 'volatility_bbhi', 
                         'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl', 
                         'volatility_dcl', 'volatility_dch', 'trend_sma_fast', 'trend_sma_slow', 
                         'trend_ema_fast', 'trend_ema_slow', 'volatility_atr', 'volume_mfi']

    missing_cols = [col for col in required_features if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    # Only scale the required features
    scaled_data = scaler.fit_transform(data[required_features])

    # Generate sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, data.columns.get_loc('close')])  # Assuming 'close' is what you're predicting

    return np.array(X), np.array(y), scaler

def visualize_data():
    """Visualize the historical and current session's predicted vs. actual prices."""
    df = load_data()
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Predicted'], label='Predicted Price', color='red')
    plt.plot(df['Date'], df['Actual'], label='Actual Price', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Stock Price Prediction vs Actual for {symbol}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_data(predicted, actual):
    """Append the predicted and actual prices to a CSV file for persistence."""
    df = pd.DataFrame({'Date': [datetime.now().strftime('%Y-%m-%d')], 'Predicted': [predicted], 'Actual': [actual]})
    if os.path.exists(data_file):
        df.to_csv(data_file, mode='a', header=False, index=False)
    else:
        df.to_csv(data_file, mode='w', header=True, index=False)

def load_data():
    """Load historical data from a CSV file."""
    if os.path.exists(data_file):
        return pd.read_csv(data_file)
    return pd.DataFrame(columns=['Date', 'Predicted', 'Actual'])

def main():
    
    sequence_length = 60 
    last_data_fetch_date = None
    historical_data = None
    processed_data = None
    scaler = None
    lstm_model = None
    in_position = False
    buy_price = 0  # Initialize buy_price to track the price at which we bought
    predicted_prices = []
    actual_prices = []

    while True:
        today = date.today()
        if last_data_fetch_date is None or last_data_fetch_date != today:
            historical_data = fetch_data(symbol)
            if not historical_data.empty:
                # Ensure data is in the correct order
                historical_data = historical_data.iloc[::-1]
                # We don't use 'processed_data' here directly because we need to split data into sequences
                X, y, scaler = preprocess_data(historical_data, sequence_length)
                # Assuming you want to predict 'close' price
                # Reshape data to fit the LSTM layer
                input_shape = (X.shape[1], X.shape[2])
                lstm_model = build_lstm_model(input_shape)
                print("Fitting model with new data")
                lstm_model.fit(X, y, epochs=10, batch_size=32)
                last_data_fetch_date = today
        
        if is_market_open():
        #if True: #for debuging while market is closed 
            current_price = fetch_current_price(symbol)
            actual_prices.append(current_price)
            if historical_data is not None and not historical_data.empty:
                latest_processed, _, _ = preprocess_data(historical_data, sequence_length)
                if latest_processed.shape[0] > 0:  # Check if we have at least one sequence
                    latest_sequence = latest_processed[-1].reshape(1, sequence_length, -1)
                    predicted_close_price = lstm_model.predict(latest_sequence)[-1, 0]
                    current_price = fetch_current_price()
                    save_data(predicted_close_price, current_price)
                    predicted_prices.append(predicted_close_price)
                    logging.info(f"Predicted price for {today.strftime('%Y-%m-%d')}: {predicted_close_price}, Actual: {current_price}")

                else:
                    print("NO DATA no sequence")
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
            sleep((30 * 60))  # Wait a defined time before the next iteration
        else:
            if today != last_data_fetch_date:
                logging.info("Market closed. Processing after-market tasks.")
                if len(predicted_prices) > 1 and len(actual_prices) > 1:
                    visualize_predictions(actual_prices, predicted_prices)            
            print("Market closed. Waiting...")
            sleep((30 * 60))  # Check every 5 minutes

if __name__ == "__main__":
    main()
