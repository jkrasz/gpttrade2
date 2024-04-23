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
            plt.show(block=False)
        else:
            print("Required columns not found in the data file.")
    else:
        print("Data file not found.")

def visualize_conditions(conditions_history):
    """Plot all conditions on a single plot for clarity and ease of analysis."""
    condition_labels = ['Volume', 'EMA', 'RSI', 'MACD', 'Bollinger', 'AI', 'Risk', 'Profit', 'Date']
    if os.path.exists(CONDITIONS_FILE):
        conditions_df = pd.read_csv(CONDITIONS_FILE, parse_dates=['Date'])
    else:
        conditions_df = pd.DataFrame(columns=condition_labels)

    # Convert the current session's conditions history into a DataFrame
    new_conditions_df = pd.DataFrame(conditions_history, columns=condition_labels)
    #new_conditions_df['Date'] = pd.to_datetime(new_conditions_df['Date'], format='%Y-%m-%d %H:%M:%S')
    new_conditions_df['Date'] = pd.to_datetime(new_conditions_df['Date'], format='%Y-%m-%d', errors='coerce')


    updated_df = pd.concat([conditions_df, new_conditions_df], ignore_index=True)
    updated_df.to_csv(CONDITIONS_FILE, index=False)  # Save the updated DataFrame back to the file

    plt.clf()  # Clear the current figure
    ax = plt.gca()  # Get current axis
    ax.xaxis_date()  # Ensure that x-axis treats data as dates
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, label in enumerate(condition_labels[:-1]):  # Exclude the 'Date' column
        ax.plot(updated_df['Date'], updated_df[label], label=label, color=colors[i % len(colors)], marker='o')

    plt.title('Trading Conditions Over Time')
    plt.xlabel('Date')
    plt.ylabel('Condition Value')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)  # Allows the plot to update without blocking
    plt.show(block=False)