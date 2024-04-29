import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import pandas as pd
import os
from config import DATA_FILE, CONDITIONS_FILE, symbol

def visualize_data(is_initialized=False):
    """Visualize the historical and current session's predicted vs. actual prices."""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
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
            plt.figure(1)  # Create a new figure for the data plot
            plt.show(block=False)
        else:
            print("Required columns not found in the data file.")
    else:
        print("Data file not found.")

def visualize_conditions(conditions_history):
    """Plot all conditions on a single plot for clarity and ease of analysis."""
    plt.clf()  # Clear the current figure

    condition_labels = [
        'Volume', 'EMA', 'RSI', 'MACD', 'Bollinger', 'ADX', 'Stochastic K',
        'AI', 'Risk', 'Profit', 'Date','Extra Column'
    ]

    if os.path.exists(CONDITIONS_FILE):
        existing_df = pd.read_csv(CONDITIONS_FILE)
        existing_df['Date'] = pd.to_datetime(existing_df['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    else:
        existing_df = pd.DataFrame(columns=condition_labels)
        existing_df['Date'] = pd.to_datetime([], errors='coerce')  # Create empty list of NaT values

    new_conditions_df = pd.DataFrame(conditions_history, columns=condition_labels)
    
    updated_df = pd.concat([existing_df, new_conditions_df], ignore_index=True)
    updated_df.to_csv(CONDITIONS_FILE, index=False)  # Save the updated DataFrame back to the file

    ax = plt.gca()  # Get current axis

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'black', 'cyan']
   
    for i, label in enumerate(condition_labels[:-1]):  # Exclude the 'Date' column
        ax.plot(updated_df['Date'].dt.strftime('%Y-%m-%d'), updated_df[label].astype(str), label=label, color=colors[i % len(colors)], marker='o')
    ax.plot(updated_df['Date'].dt.strftime('%Y-%m-%d'), updated_df['Date'].dt.strftime('%Y-%m-%d'), label='Date', color='black', marker='o')  # Add the Date plot

    plt.title('Trading Conditions Over Time')
    plt.xlabel('Date')
    plt.ylabel('Condition Value')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.figure(2)  # Create a new figure for the data plot
    plt.show(block=False)
    plt.pause(0.1)
