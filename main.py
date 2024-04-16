from datetime import date
from time import sleep
import pandas as pd
import matplotlib.pyplot as plt
import os
from logger_config import setup_logging
from data_manager import fetch_data, fetch_current_price
from trading_model import build_lstm_model, predict_price
from visualization import visualize_data, visualize_conditions
from email_notifications import send_email
from trading_strategy import should_buy, should_sell
from config import symbol, DATA_FILE, CONDITIONS_FILE
from data_manager import fetch_data, fetch_current_price, preprocess_data, is_market_open,save_data, load_data

def main():
    plt.ion()  # Enable interactive mode for plot updates
    logger = setup_logging()
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

    visualize_data(is_initialized)

    # Initialize conditions history from file if it exists
    if os.path.exists(CONDITIONS_FILE):
        conditions_history = pd.read_csv(CONDITIONS_FILE).values.tolist()
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
                    visualize_data(is_initialized)
                    is_initialized = True  # The plot is now initialized

        if is_market_open() #or True:  # True is for testing without real-time market data
            current_price = fetch_current_price(symbol)
            actual_prices.append(current_price)
            if historical_data is not None and not historical_data.empty:
                latest_processed, _, _ = preprocess_data(historical_data, sequence_length)
                if latest_processed.shape[0] > 0:  # Check if we have at least one sequence
                    latest_sequence = latest_processed[-1].reshape(1, sequence_length, -1)
                    predicted_close_price = predict_price(lstm_model, historical_data['close'].values, scaler, sequence_length)
                    current_price = fetch_current_price(symbol)
                    save_data(predicted_close_price, current_price)
                    predicted_prices.append(predicted_close_price)
                    logger.info(f"Predicted price for {today.strftime('%Y-%m-%d')}: {predicted_close_price}, Actual: {current_price}")

                avg_volume = historical_data['volume'].rolling(window=20).mean().iloc[-1]
                ema_short = historical_data['close'].ewm(span=12, adjust=False).mean().iloc[-1]
                ema_long = historical_data['close'].ewm(span=26, adjust=False).mean().iloc[-1]

                buy_signal, conditions = should_buy(predicted_close_price, historical_data.iloc[-1].to_dict(), avg_volume, ema_short, ema_long, current_price=current_price)
                conditions_history.append([int(val) for val in conditions]) 
                visualize_conditions(conditions_history)

                if not in_position and buy_signal:
                    in_position = True
                    buy_price = current_price
                    print(f"Buy at {buy_price}")
                    send_email("Buy Signal", f"Buy at price {buy_price}")
                elif in_position and should_sell(historical_data.iloc[-1].to_dict(), buy_price, current_price=current_price):
                    in_position = False
                    sell_price = current_price
                    print(f"Sell at {sell_price}")
                    send_email("Sell Signal", f"Sell at price {sell_price}")
                    buy_price = 0  # Reset buy_price after selling
            else:
                print("Historical data is empty. Skipping this cycle.")

            print("Wait for the next iteration")
            sleep(60 * 60)  # Wait a defined time before the next iteration
        else:
            if today != last_data_fetch_date:
                logger.info("Market closed. Processing after-market tasks.")
            print("Market closed. Waiting...")
            sleep(60 * 60)  # Check every hour

if __name__ == "__main__":
    main()
