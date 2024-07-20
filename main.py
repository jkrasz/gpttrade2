from datetime import date, datetime
from time import sleep
import pandas as pd
import os
from logger_config import setup_logging
from data_manager import fetch_data, fetch_current_price, preprocess_data, is_market_open, save_data, load_data
from trading_model import train_models, predict_price
from visualization import visualize_data, visualize_conditions
from email_notifications import send_email
from trading_strategy import should_buy, should_sell
from config import symbol, DATA_FILE, CONDITIONS_FILE
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    logger = setup_logging()
    sequence_length = 120
    last_data_fetch_date = None
    historical_data = None
    scaler = None
    models = None
    in_position = False
    buy_price = 0
    predicted_prices = []
    actual_prices = []
    conditions_history = []
    visualize_data()

    while True:
        today = date.today()
        
        if last_data_fetch_date is None or last_data_fetch_date != today:
            historical_data = fetch_data(symbol)
            if not historical_data.empty:
                X, y, scaler = preprocess_data(historical_data, sequence_length)
                input_shape = (X.shape[1], X.shape[2])
                
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
                
                models = train_models(X_train, y_train, input_shape, epochs=50, batch_size=32)
                
                for i, model in enumerate(models[:4]):
                    val_loss = model.evaluate(X_val, y_val, verbose=0)
                    logger.info(f"Model {i+1} validation loss: {val_loss}")
                
                last_data_fetch_date = today

        if is_market_open():# or True:
            current_price = fetch_current_price(symbol)
            actual_prices.append(current_price)
            if historical_data is not None and not historical_data.empty and models is not None:                        
                historical_data_slice = historical_data.iloc[-sequence_length:]
                print(f"Number of rows in historical_data_slice: {len(historical_data_slice)}")
                latest_processed, _, _ = preprocess_data(historical_data_slice, sequence_length)
                print(f"latest_processed shape: {latest_processed.shape}")
                
                if latest_processed.shape[0] > 0:
                    if latest_processed.shape[0] == 1:
                        # Use the single sequence for prediction
                        predicted_close_price = predict_price(models, latest_processed[0], scaler, sequence_length)
                    else:
                        # Use the last sequence for prediction
                        predicted_close_price = predict_price(models, latest_processed[-1], scaler, sequence_length)
                    
                    if predicted_close_price is not None:
                        save_data(predicted_close_price, current_price)
                        predicted_prices.append(predicted_close_price)
                        logger.info(f"Predicted price for {today.strftime('%Y-%m-%d')}: {predicted_close_price}, Actual: {current_price}")
                        
    
                    # Calculate technical indicators
                    avg_volume = historical_data['volume'].rolling(window=20).mean().iloc[-1]
                    ema_short = historical_data['close'].ewm(span=12, adjust=False).mean().iloc[-1]
                    ema_long = historical_data['close'].ewm(span=26, adjust=False).mean().iloc[-1]
                    sma_50 = historical_data['close'].rolling(window=50).mean().iloc[-1]
                    sma_200 = historical_data['close'].rolling(window=200).mean().iloc[-1]
                    rsi = historical_data['close'].rolling(window=14).apply(lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).sum() / abs(x.diff().clip(upper=0).sum())))))

                    buy_signal, condition_values = should_buy(predicted_close_price, historical_data.iloc[-1].to_dict(), 
                                                              avg_volume, ema_short, ema_long, current_price=current_price,
                                                              sma_50=sma_50, sma_200=sma_200, rsi=rsi.iloc[-1])

                    conditions_history.append(condition_values)
                    visualize_conditions(conditions_history)

                    if not in_position and buy_signal:
                        in_position = True
                        buy_price = current_price
                        logger.info(f"Buy signal triggered at {buy_price}")
                        send_email("Buy Signal", f"Buy at price {buy_price}")
                    elif in_position and should_sell(historical_data.iloc[-1].to_dict(), buy_price, current_price=current_price):
                        in_position = False
                        sell_price = current_price
                        profit_loss = (sell_price - buy_price) / buy_price * 100
                        logger.info(f"Sell signal triggered at {sell_price}. Profit/Loss: {profit_loss:.2f}%")
                        send_email("Sell Signal", f"Sell at price {sell_price}. Profit/Loss: {profit_loss:.2f}%")
                        buy_price = 0  # Reset buy_price after selling

                    if len(actual_prices) % 30 == 0:
                        logger.info("Retraining models...")
                        X, y, scaler = preprocess_data(historical_data, sequence_length)
                        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
                        models = train_models(X_train, y_train, input_shape, epochs=50, batch_size=32)
                else:
                    logger.warning("Processed data is empty. Skipping this cycle.")
            else:
                logger.warning("Historical data is empty or models are not trained. Skipping this cycle.")

            logger.info("Wait for the next iteration")
            sleep(45 * 60)
        else:
            if today != last_data_fetch_date:
                logger.info("Market closed. Processing after-market tasks.")
            logger.info("Market closed. Waiting...")
            sleep(60 * 60)

if __name__ == "__main__":
    main()
