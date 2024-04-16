import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import os
import pandas as pd  # Ensure pandas is also imported if it's used
from config import DATA_FILE, symbol ,CONDITIONS_FILE # Import symbol if used for the plot title

def visualize_data(is_initialized=False):
    """Visualize the historical and current session's predicted vs. actual prices."""
    if os.path.exists(DATA_FILE):  # Correct variable name used here
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
        else:
            print("Required columns not found in the data file.")
    else:
        print("Data file not found.")



def visualize_conditions(conditions_history):
    condition_labels = ['Volume', 'EMA', 'RSI', 'MACD', 'Bollinger', 'AI', 'Risk', 'Profit']

    if os.path.exists(CONDITIONS_FILE):
        # Load existing conditions data from file
        existing_df = pd.read_csv(CONDITIONS_FILE)
    else:
        existing_df = pd.DataFrame(columns=condition_labels)

    # Convert the current session's conditions history into a DataFrame
    conditions_df = pd.DataFrame(conditions_history, columns=condition_labels).astype(int)

    # Append the new conditions to the existing DataFrame
    updated_df = pd.concat([existing_df, conditions_df], ignore_index=True)

    # Save the updated DataFrame back to the file
    updated_df.to_csv(CONDITIONS_FILE, index=False)

    # Now plot the updated DataFrame on a single plot
    plt.clf()  # Clear the current figure
    ax = plt.gca()  # Get current axis

    # Plot each condition with a unique color
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, label in enumerate(condition_labels):
        ax.plot(updated_df.index, updated_df[label], label=label, color=colors[i % len(colors)], marker='o')

    plt.title('Buy Signal Conditions Over Time')
    plt.xlabel('Index')
    plt.ylabel('Condition Value')
    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)  # Allows the plot to update without blocking
    plt.show(block=False)