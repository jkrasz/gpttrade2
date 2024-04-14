import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
LOG_FILE = 'stock_predictions.log'
DATA_FILE = 'historical_data.csv'
CONDITIONS_FILE = 'conditions_data.csv'
EMAIL_USER = 'chatGptTrade@gmail.com'
EMAIL_PASS = 'bymwlvzzmbzxeeas'
RECEIVER_EMAIL = 'john.kraszewski@gmail.com'
symbol = 'GPRO'

# Ensure the logging directory exists
os.makedirs('logs', exist_ok=True)
