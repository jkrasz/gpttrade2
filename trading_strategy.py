from logger_config import setup_logging
import numpy as np
logger = setup_logging()
from datetime import datetime

def should_buy(predicted_close_price, current_data, average_volume, ema_short, ema_long,
               threshold_rsi_buy=30, threshold_volume_increase=1.2,
               macd_signal_threshold=0, bollinger_band_window=20,
               bollinger_band_std_dev=2, current_price=None, price_jump_threshold=1.03,
               risk_tolerance=0.05, profit_tolerance=0.10,
               adx_threshold=25, stochastic_k_threshold=20):
    
    # Initialize logging
    logger = setup_logging()

    # Extract metrics from current_data with defaults
    rsi = current_data.get('momentum_rsi', 100)
    macd = current_data.get('trend_macd', 0)
    macd_signal = current_data.get('trend_macd_signal', 0)
    volume = current_data.get('volume', 0)
    close_price = current_price if current_price is not None else current_data.get('close', 0)
    bb_lower_band = current_data.get('volatility_bbl', np.inf)
    adx = current_data.get('trend_adx', 0)
    stochastic_k = current_data.get('momentum_stoch_k', 100)
    current_date = datetime.now()

    # Condition checks
    condition_dict = {
        'Volume': volume > average_volume * threshold_volume_increase,
        'EMA': ema_short > ema_long,
        'RSI': rsi < threshold_rsi_buy,
        'MACD': (macd > macd_signal) and (macd > macd_signal_threshold),
        'Bollinger': close_price <= bb_lower_band,
        'ADX': adx > adx_threshold,
        'Stochastic K': stochastic_k < stochastic_k_threshold,
        'AI': False,  # Placeholder for AI, adjust as necessary
        'Risk': predicted_close_price > close_price * (1 - risk_tolerance),
        'Profit': predicted_close_price > close_price * (1 + profit_tolerance),
        'Date': current_date.strftime('%Y-%m-%d %H:%M:%S')  # Format for consistency
    }

    buy_signal = all(condition_dict.values())

    logger.info("Trading conditions checked: %s", condition_dict)
    logger.info("Buy signal: %s", buy_signal)

    return buy_signal, condition_dict 


# Sell conditions function
def should_sell(current_data, buy_price, stop_loss_percent=0.10, take_profit_percent=0.15, threshold_rsi_sell=70, current_price=None):
    # Use provided current price if available, otherwise fall back to the last known close price
    current_price = current_price if current_price is not None else current_data.get('close', 0)
    rsi = current_data.get('momentum_rsi', 0)
    sell_signal = current_price <= buy_price * (1 - stop_loss_percent) or \
                  current_price >= buy_price * (1 + take_profit_percent) or \
                  rsi > threshold_rsi_sell
    return sell_signal
