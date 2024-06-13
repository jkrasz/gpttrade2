import pandas as pd
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from config import CONDITIONS_FILE, DATA_FILE, symbol
import json

global fig_conditions
fig_conditions = None
global fig_data
fig_data = None

def visualize_data():
    """Visualize the historical and current session's predicted vs. actual prices with Plotly, updating the plot with new data."""
    global fig_data
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        if 'Date' in df.columns and 'Predicted' in df.columns and 'Actual' in df.columns:
            if fig_data is None:
                fig_data = make_subplots(rows=1, cols=1)
                fig_data.add_trace(go.Scatter(x=df['Date'], y=df['Predicted'], mode='lines', name='Predicted Price'), row=1, col=1)
                fig_data.add_trace(go.Scatter(x=df['Date'], y=df['Actual'], mode='lines', name='Actual Price'), row=1, col=1)
                fig_data.update_layout(title=f'Stock Price Prediction vs Actual for {symbol}',
                                       xaxis_title='Date',
                                       yaxis_title='Price')
            else:
                # Update existing traces
                fig_data.data[0].x = df['Date']
                fig_data.data[0].y = df['Predicted']
                fig_data.data[1].x = df['Date']
                fig_data.data[1].y = df['Actual']
            pio.show(fig_data, auto_open=False)
        else:
            print("Required columns not found in the data file.")
    else:
        print("Data file not found.")

def visualize_conditions(condition_values):
    """Plot all conditions on a single plot for clarity and ease of analysis with Plotly, updating the plot with new data."""
    global fig_conditions
    if os.path.exists(CONDITIONS_FILE):
        existing_df = pd.read_csv(CONDITIONS_FILE)
        existing_df['Date'] = pd.to_datetime(existing_df['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    else:
        existing_df = pd.DataFrame()

    # Ensure condition_values is a list of dictionaries
    if not isinstance(condition_values, list):
        condition_values = [condition_values]

    new_conditions_df = pd.DataFrame(condition_values)
    updated_df = pd.concat([existing_df, new_conditions_df], ignore_index=True)
    updated_df = updated_df.drop_duplicates(subset='Date', keep='last')
    updated_df.to_csv(CONDITIONS_FILE, index=False)

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'black', 'cyan']

    if fig_conditions is None:
        fig_conditions = make_subplots(rows=1, cols=1)
        for i, label in enumerate(new_conditions_df.columns[:-1]):  # Skip 'Date' column for looping
            if label in updated_df.columns:
                fig_conditions.add_trace(go.Scatter(x=updated_df['Date'], y=updated_df[label], mode='lines+markers',
                                                    name=label, line=dict(color=colors[i % len(colors)])), row=1, col=1)

        fig_conditions.update_layout(title='Trading Conditions Over Time',
                                     xaxis_title='Date',
                                     yaxis_title='Condition Value',
                                     legend_title="Conditions",
                                     legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1))
    else:
        for i, trace in enumerate(fig_conditions['data']):
            trace.x = updated_df['Date']
            trace.y = updated_df[trace.name]

    pio.show(fig_conditions, auto_open=False)
