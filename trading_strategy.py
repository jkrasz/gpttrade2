from logger_config import setup_logging
import numpy as np
logger = setup_logging()

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
    logger.info(f"Condition values: {condition_values}")

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
