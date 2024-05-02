import pandas as pd
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from config import DATA_FILE, CONDITIONS_FILE, symbol

def visualize_data():
    """Visualize the historical and current session's predicted vs. actual prices with Plotly."""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        if 'Date' in df.columns and 'Predicted' in df.columns and 'Actual' in df.columns:
            fig = make_subplots(rows=1, cols=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Predicted'], mode='lines', name='Predicted Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Actual'], mode='lines', name='Actual Price'), row=1, col=1)

            fig.update_layout(title=f'Stock Price Prediction vs Actual for {symbol}',
                              xaxis_title='Date',
                              yaxis_title='Price')
            fig.show()
        else:
            print("Required columns not found in the data file.")
    else:
        print("Data file not found.")

def visualize_conditions(conditions_history):
    """Plot all conditions on a single plot for clarity and ease of analysis with Plotly."""
    if os.path.exists(CONDITIONS_FILE):
        existing_df = pd.read_csv(CONDITIONS_FILE)
        existing_df['Date'] = pd.to_datetime(existing_df['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    else:
        existing_df = pd.DataFrame()

    new_conditions_df = pd.DataFrame(conditions_history)
    updated_df = pd.concat([existing_df, new_conditions_df], ignore_index=True)
    updated_df = updated_df.drop_duplicates(subset='Date', keep='last')
    updated_df.to_csv(CONDITIONS_FILE, index=False)

    fig = make_subplots(rows=1, cols=1)
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'black', 'cyan']

    for label in new_conditions_df.columns[:-1]:  # Skip 'Date' column for looping
        if label in updated_df.columns:
            fig.add_trace(go.Scatter(x=updated_df['Date'], y=updated_df[label], mode='lines+markers',
                                     name=label, line=dict(color=colors[int(updated_df.columns.get_loc(label) % len(colors))])), row=1, col=1)

    fig.update_layout(title='Trading Conditions Over Time',
                      xaxis_title='Date',
                      yaxis_title='Condition Value',
                      legend_title="Conditions",
                      legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1))
    fig.show()

