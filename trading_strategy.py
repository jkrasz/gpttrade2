from logger_config import setup_logging
import numpy as np
from datetime import datetime

logger = setup_logging()

def should_buy(predicted_close_price, current_data, average_volume, ema_short, ema_long,
               threshold_rsi_buy=40, threshold_volume_increase=1.2,
               macd_signal_threshold=0, bollinger_band_window=20,
               bollinger_band_std_dev=2, current_price=None, price_jump_threshold=1.05,
               risk_tolerance=0.15, profit_tolerance=0.25,
               adx_threshold=25, stochastic_k_threshold=20,
               sma_50=None, sma_200=None, rsi=None):

    rsi = current_data.get('momentum_rsi', 100)
    macd = current_data.get('trend_macd', 0)
    macd_signal = current_data.get('trend_macd_signal', 0)
    volume = current_data.get('volume', 0)
    close_price = current_price if current_price is not None else current_data.get('close', 0)
    bb_lower_band = current_data.get('volatility_bbl', np.inf)
    adx = current_data.get('trend_adx', 0)
    stochastic_k = current_data.get('momentum_stoch_k', 100)
    golden_cross = sma_50 > sma_200 if sma_50 and sma_200 else False
    rsi_condition = 30 <= rsi <= 70 if rsi is not None else False
    current_date = datetime.now()

    # Ensure predicted_close_price is not None
    if predicted_close_price is None:
        logger.error("Predicted close price is None, skipping buy signal evaluation.")
        return False, {}

    condition_dict = {
        'Volume': volume > average_volume * threshold_volume_increase,
        'EMA': ema_short > ema_long,
        'RSI': rsi_condition,
        'MACD': (macd > macd_signal) and (macd > macd_signal_threshold),
        'Bollinger': close_price <= bb_lower_band,
        'ADX': adx > adx_threshold,
        'Stochastic K': stochastic_k < stochastic_k_threshold,
        'AI': predicted_close_price > close_price * 1.1,
        'Risk': predicted_close_price > close_price * (1 - risk_tolerance),
        'Profit': predicted_close_price > close_price * (1 + profit_tolerance),
        'Price Jump': predicted_close_price > close_price * price_jump_threshold,
        'Golden Cross': golden_cross,
    }

    # Count the number of True conditions
    true_conditions = sum(1 for condition in condition_dict.values() if condition)
    
    # Add the date to the condition_dict, but don't include it in the buy signal calculation
    condition_dict['Date'] = current_date.strftime('%Y-%m-%d %H:%M:%S')

    buy_signal = true_conditions >= 10
    print(f"Number of true conditions: {true_conditions}")
    print(f"Buy signal: {buy_signal}")
    logger.info("Trading conditions checked: %s", condition_dict)
    logger.info("Buy signal: %s", buy_signal)

    return buy_signal, condition_dict

def should_sell(current_data, buy_price, stop_loss_percent=0.07, take_profit_percent=0.20, threshold_rsi_sell=65, current_price=None):
    # Use provided current price if available, otherwise fall back to the last known close price
    current_price = current_price if current_price is not None else current_data.get('close', 0)
    rsi = current_data.get('momentum_rsi', 0)
    sell_signal = current_price <= buy_price * (1 - stop_loss_percent) or \
                  current_price >= buy_price * (1 + take_profit_percent) or \
                  rsi > threshold_rsi_sell
    return sell_signal
