# logger_config.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    """
    Set up the logging configuration.
    """
    logger = logging.getLogger('TradingLogger')
    logger.setLevel(logging.INFO)

    # Create handlers
    handler = RotatingFileHandler('trading_system.log', maxBytes=10000, backupCount=3)
    handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(handler)
    return logger
