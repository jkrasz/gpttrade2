
import pandas as pd
import numpy as np
import yfinance as yf
from stable_baselines3 import PPO
from custom_env import StockTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ta
from datetime import datetime, timedelta, time
import pytz
from time import sleep
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import os
from dotenv import load_dotenv


# Market open check function
def is_market_open():
    opening_time = time(9, 30)
    closing_time = time(16, 0)
    now = datetime.now(pytz.timezone('US/Eastern'))
    return opening_time <= now.time() <= closing_time and now.weekday() < 5

# Send email function
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

# Fetch and process data function
def fetch_data(symbol='GPRO'):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Use last year's data for training
    data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    data.replace(0, np.nan, inplace=True)
    data = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    data['RSI'] = ta.momentum.rsi(data['Close']).fillna(50)  # Replace NaN values with a neutral RSI value like 50
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['RSI'] = ta.momentum.rsi(data['Close'])
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
    data['EMA'] = ta.trend.ema_indicator(data['Close'])
    data['MFI'] = ta.volume.money_flow_index(data['High'], data['Low'], data['Close'], data['Volume'])
    macd_indicator = ta.trend.MACD(data['Close'])
    data['MACD_Line'] = macd_indicator.macd()
    data['Signal_Line'] = macd_indicator.macd_signal()
    data['RSI_Avg'] = data['RSI'].rolling(window=5).mean()
    data['Volatility'] = data['Close'].rolling(window=20).std()
    data.dropna(inplace=True)
    # Ensure 'RSI' is in DataFrame
    print(data.columns)  # Debugging: Verify that 'RSI' is a column name

    return data

# Build LSTM model function
def build_lstm_model(input_shape, units=50, dropout_rate=0.2):
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units=units),
        Dropout(dropout_rate),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Preprocess data for LSTM
def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Close', 'Volume', 'momentum_rsi', 'trend_macd', 'trend_macd_signal']])
    return scaled_data

# Dynamic stop loss and take profit
def dynamic_stop_loss_percent(volatility):
    return max(0.90, 1 - volatility / 100)

def dynamic_take_profit_percent(volatility):
    return min(1.15, 1 + volatility / 50)

# Main function
def main():
    symbol = 'GPRO'
    threshold_rsi_sell = 70
    stop_loss_percent = 0.95
    take_profit_percent = 1.1
    threshold_rsi_buy = 30
    actions = []
    in_position = False
    data = fetch_data(symbol)
    average_volatility = data['volatility_atr'].mean()
    processed_data = preprocess_data(data)
    lstm_model = build_lstm_model((processed_data.shape[1], 1))
    lstm_model.fit(processed_data, data['Close'], epochs=10, batch_size=32)
    env = DummyVecEnv([lambda: StockTradingEnv(data)])
    model = PPO("MlpPolicy", env, verbose=1)

    while True:
        #if True:
        if is_market_open():
            current_data = fetch_data()  # Efficiently fetch and preprocess only new data
            processed_current_data = preprocess_data(current_data)
            lstm_predictions = lstm_model.predict(processed_current_data)
            model.learn(total_timesteps=20000)
            last_data_point_index = len(data) - 1
            obs = env.reset()

            for i in range(len(data)):
                close_price = data.iloc[i]['Close']
                rsi = data.iloc[i]['RSI']
                macd = data.iloc[i]['MACD_Line']
                signal = data.iloc[i]['Signal_Line']
                volatility = data.iloc[i]['Volatility']
                ema_short = data.iloc[i]['EMA']  # Assuming you have this column after preprocessing
                ema_long = data.iloc[i]['SMA_20']  # Assuming this represents a longer period EMA
                
                # Adjust thresholds based on volatility
                if volatility > average_volatility:  # Assuming you've calculated average_volatility beforehand
                    threshold_rsi_buy += 5  # Making it harder to buy in high volatility
                    threshold_rsi_sell -= 5  # Easier to sell
                else:
                    threshold_rsi_buy -= 5  # Easier to buy in low volatility
                    threshold_rsi_sell += 5  # Harder to sell

                # Trend confirmation
                trend_up = ema_short > ema_long
                trend_down = ema_short < ema_long

                # Dynamic stop-loss and take-profit
                stop_loss_percent = dynamic_stop_loss_percent(volatility)
                take_profit_percent = dynamic_take_profit_percent(volatility)
                
                # Position sizing (example: risking 1% of account balance per trade)
                account_balance = 10000  # Example account balance
                risk_per_trade = 0.01
                trade_risk = account_balance * risk_per_trade
                position_size = trade_risk / (close_price * stop_loss_percent)

                if not in_position and rsi < threshold_rsi_buy and macd > signal and trend_up:
                    in_position = True
                    buy_price = close_price
                    actions.append((i, "Buy", close_price, position_size))
                    if i == last_data_point_index:
                        send_email("Buy Signal", f"Buy {position_size} units at price {close_price}")
                elif in_position:
                    if (rsi > threshold_rsi_sell or macd < signal or close_price <= buy_price * stop_loss_percent or close_price >= buy_price * take_profit_percent) and trend_down:
                        in_position = False
                        actions.append((i, "Sell", close_price, position_size))
                        if i == last_data_point_index:
                            send_email("Sell Signal", f"Sell {position_size} units at price {close_price}")

            print("Simulation Completed! Sleeping...")
            sleep(60 * 15)  # Efficient use of sleep based on market status
        else:
            print("Market is closed. Sleeping...")
            sleep(60 * 15)

if __name__ == "__main__":
    main()