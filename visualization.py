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
    plt.clf()

    if os.path.exists(CONDITIONS_FILE):
        existing_df = pd.read_csv(CONDITIONS_FILE)
        existing_df['Date'] = pd.to_datetime(existing_df['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    else:
        existing_df = pd.DataFrame()

    # Assume conditions_history is a list of dictionaries
    new_conditions_df = pd.DataFrame(conditions_history)
    
    updated_df = pd.concat([existing_df, new_conditions_df], ignore_index=True)
    updated_df = updated_df.drop_duplicates(subset='Date', keep='last')
    updated_df.to_csv(CONDITIONS_FILE, index=False)

    ax = plt.gca()
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'black', 'cyan']
    for label in new_conditions_df.columns[:-1]:  # Skip 'Date' column for looping
        if label in updated_df.columns:
            ax.plot(updated_df['Date'], updated_df[label], label=label, color=colors[int(updated_df.columns.get_loc(label) % len(colors))], marker='o')

    plt.title('Trading Conditions Over Time')
    plt.xlabel('Date')
    plt.ylabel('Condition Value')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)