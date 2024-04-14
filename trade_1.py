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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Initialization
load_dotenv()
logging.basicConfig(filename='stock_predictions.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
data_file = 'historical_data.csv'
conditions_file = 'conditions_data.csv'
# Ensure the logging directory exists
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/stock_predictions.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set API key and symbol
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
symbol = 'GPRO'

# Ensure logging directory exists
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/stock_predictions.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def send_email(subject, content):
    sender_email = 'chatGptTrade@gmail.com'
    sender_password = 'bymwlvzzmbzxeeas'  # In a real scenario, use a secure method to store and access credentials
    receiver_email = 'john.kraszewski@gmail.com'
    smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
    smtp_server.starttls()
    smtp_server.login(sender_email, sender_password)
    message = f"""Subject: {subject}\n\n{content}"""
    print(message)
    try:
        smtp_server.sendmail(sender_email, receiver_email, message)
    except Exception as e:
        print("Error sending email: ", e)
    smtp_server.quit()

def fetch_current_price(symbol='GPRO'):
    ts = TimeSeries(api_key, output_format='pandas')
    data, _ = ts.get_quote_endpoint(symbol)
    if '05. price' in data.columns:
        return float(data['05. price'].iloc[0])  # Updated to avoid FutureWarning
    raise ValueError("Current price not found in response.")

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

    condition_values = [
        volume_condition,
        ema_condition,
        rsi_condition,
        macd_condition,
        bollinger_condition,
        ai_condition,
        risk_condition,
        profit_condition
    ]

    buy_signal = all(condition_values)
    logging.info(f"Condition values: {condition_values}")

    return buy_signal, condition_values

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

def visualize_data(is_initialized=False):
    """Visualize the historical and current session's predicted vs. actual prices."""
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        if 'Date' in df.columns and 'Predicted' in df.columns and 'Actual' in df.columns:
            if not is_initialized:
                plt.ion()  # Turn on interactive mode
                fig, ax = plt.subplots(figsize=(12, 6))
            else:
                plt.clf()  # Clear the current figure
                fig, ax = plt.subplots(figsize=(12, 6))

            ax.plot(df['Date'], df['Predicted'], label='Predicted Price', color='red')
            ax.plot(df['Date'], df['Actual'], label='Actual Price', color='blue')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title(f'Stock Price Prediction vs Actual for {symbol}')
            plt.xticks(rotation=45)
            ax.legend()
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)  # Allows the plot to update without blocking
        else:
            print("Required columns not found in the data file.")
    else:
        print("Data file not found.")


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

def preprocess_data(data, sequence_length=60):
    """
    Preprocess the data
    """
    # Selecting the 'close' prices only
    close_prices = data['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

def predict_price(model, data, scaler, sequence_length=60):
    """
    Make a prediction for the next closing price.
    """
    # Ensure the last sequence is a 2D array as expected by scaler
    last_sequence = data[-sequence_length:].reshape(-1, 1)  # Reshaping to 2D array
    last_sequence_scaled = scaler.transform(last_sequence)
    last_sequence_scaled = np.reshape(last_sequence_scaled, (1, sequence_length, 1))
    predicted_scaled = model.predict(last_sequence_scaled)
    predicted_price = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))[0, 0]
    
    return predicted_price

def visualize_conditions(conditions_history):
    condition_labels = ['Volume', 'EMA', 'RSI', 'MACD', 'Bollinger', 'AI', 'Risk', 'Profit']

    if os.path.exists(conditions_file):
        # Load existing conditions data from file
        existing_df = pd.read_csv(conditions_file)
    else:
        existing_df = pd.DataFrame(columns=condition_labels)

    # Convert the current session's conditions history into a DataFrame
    conditions_df = pd.DataFrame(conditions_history, columns=condition_labels).astype(int)

    # Append the new conditions to the existing DataFrame
    updated_df = pd.concat([existing_df, conditions_df], ignore_index=True)

    # Save the updated DataFrame back to the file
    updated_df.to_csv(conditions_file, index=False)

    # Now plot the updated DataFrame
    plt.clf()  # Clear the current figure
    updated_df.plot(subplots=True, layout=(4, 2), figsize=(15, 10), marker='o', title='Buy Signal Conditions Over Time')
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)  # Allows the plot to update without blocking



def main():
    plt.ion()  # Enable interactive mode for plot updates
    is_initialized = False
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
    visualize_data()
    if os.path.exists(conditions_file):
        conditions_history = pd.read_csv(conditions_file).values.tolist()
    else:
        conditions_history = []
        
    while True:
        today = date.today()
        
        if last_data_fetch_date is None or last_data_fetch_date != today:
            historical_data = fetch_data(symbol)
            if not historical_data.empty:
                historical_data = historical_data.iloc[::-1]
                X, y, scaler = preprocess_data(historical_data, sequence_length)
                input_shape = (X.shape[1], X.shape[2])
                lstm_model = build_lstm_model(input_shape)
                print("Fitting model with new data")
                lstm_model.fit(X, y, epochs=20, batch_size=32)  # Increase the number of epochs
                last_data_fetch_date = today
                if len(predicted_prices) > 1 and len(actual_prices) > 1:
                    visualize_data( is_initialized)
                    is_initialized = True  # The plot is now initialized

        if is_market_open() or True:  # True is for testing without real-time market data
            current_price = fetch_current_price(symbol)
            actual_prices.append(current_price)
            if historical_data is not None and not historical_data.empty:
                latest_processed, _, _ = preprocess_data(historical_data, sequence_length)
                if latest_processed.shape[0] > 0:  # Check if we have at least one sequence
                    latest_sequence = latest_processed[-1].reshape(1, sequence_length, -1)
                    predicted_close_price = predict_price(lstm_model, historical_data['close'].values, scaler, sequence_length)
                    current_price = fetch_current_price()
                    save_data(predicted_close_price, current_price)
                    predicted_prices.append(predicted_close_price)
                    logging.info(f"Predicted price for {today.strftime('%Y-%m-%d')}: {predicted_close_price}, Actual: {current_price}")

                else:
                    print("NO DATA no sequence")
                avg_volume = historical_data['volume'].rolling(window=20).mean().iloc[-1]
                ema_short = historical_data['close'].ewm(span=12, adjust=False).mean().iloc[-1]
                ema_long = historical_data['close'].ewm(span=26, adjust=False).mean().iloc[-1]

                buy_signal, conditions = should_buy(predicted_close_price, historical_data.iloc[-1].to_dict(), avg_volume, ema_short, ema_long, threshold_rsi_buy=30, threshold_volume_increase=1.2, macd_signal_threshold=0, bollinger_band_window=20, bollinger_band_std_dev=2, current_price=current_price, price_jump_threshold=1.05, risk_tolerance=0.03, profit_tolerance=0.12)
                conditions_history.append([int(val) for val in conditions]) 
                visualize_conditions(conditions_history)

                # Use current price and historical data to decide whether to buy or sell
                if not in_position and buy_signal:
                    in_position = True
                    buy_price = current_price
                    print(f"Buy at {buy_price}")
                    send_email("Buy Signal", f"Buy at price {buy_price}")
                elif in_position and should_sell(historical_data.iloc[-1].to_dict(), buy_price, stop_loss_percent=0.08, take_profit_percent=0.18, threshold_rsi_sell=70, current_price=current_price):  # Adjust the stop_loss_percent and take_profit_percent
                    in_position = False
                    sell_price = current_price
                    print(f"Sell at {sell_price}")
                    buy_price = 0  # Reset buy_price after selling
            else:
                print("Historical data is empty. Skipping this cycle.")

            print("Wait for the next iteration")
            sleep((60 * 60))  # Wait a defined time before the next iteration
        else:
            if today != last_data_fetch_date:
                logging.info("Market closed. Processing after-market tasks.")
                
            print("Market closed. Waiting...")
            sleep((60 * 60))  # Check every 5 minutes

if __name__ == "__main__":
    main()
