import pandas as pd
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from config import CONDITIONS_FILE, DATA_FILE, symbol

global fig_conditions
fig_conditions = None
global fig_data
fig_data = None

def initialize_figures():
    """Initialize the global figures for data and conditions plots."""
    global fig_conditions, fig_data
    fig_conditions = make_subplots(rows=2, cols=1, subplot_titles=('Boolean Conditions', 'Numeric Conditions'),
                                   vertical_spacing=0.1, row_heights=[0.7, 0.3])
    fig_data = make_subplots(rows=1, cols=1)
    
    # Set up initial layout for fig_data
    fig_data.update_layout(title=f'Stock Price Prediction vs Actual for {symbol}',
                           xaxis_title='Date',
                           yaxis_title='Price')
    
    # Initialize empty traces for predicted and actual prices
    fig_data.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Predicted Price'))
    fig_data.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Actual Price'))
    
    # Initialize empty traces for conditions
    boolean_columns = ['Volume', 'EMA', 'RSI', 'MACD', 'Bollinger', 'ADX', 'Stochastic K', 'AI', 'Risk', 'Profit', 'Price Jump', 'Golden Cross']
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'black', 'cyan', 
              'magenta', 'lime', 'teal', 'lavender', 'yellow', 'salmon']
    
    for i, label in enumerate(boolean_columns):
        fig_conditions.add_trace(go.Scatter(x=[], y=[], mode='lines+markers',
                                            name=label, line=dict(color=colors[i % len(colors)])), row=1, col=1)
    
    fig_conditions.update_layout(title='Trading Conditions Over Time',
                                 xaxis_title='Date',
                                 yaxis_title='Condition Value',
                                 legend_title="Conditions",
                                 legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
                                 height=800)
    
    fig_conditions.update_yaxes(title_text="Boolean Value", row=1, col=1)
    fig_conditions.update_yaxes(title_text="Numeric Value", row=2, col=1)

    # Load historical data and update plots
    load_and_update_historical_data()

def load_and_update_historical_data():
    """Load historical data from CSV files and update both plots."""
    global fig_conditions, fig_data

    # Load conditions data
    if os.path.exists(CONDITIONS_FILE):
        conditions_df = pd.read_csv(CONDITIONS_FILE)
        # Ensure 'Date' is the last column
        conditions_df = conditions_df[[col for col in conditions_df.columns if col != 'Date'] + ['Date']]
        
        dates = conditions_df['Date'].tolist()
        boolean_columns = ['Volume', 'EMA', 'RSI', 'MACD', 'Bollinger', 'ADX', 'Stochastic K', 'AI', 'Risk', 'Profit', 'Price Jump', 'Golden Cross']
        numeric_columns = [col for col in conditions_df.columns if col not in boolean_columns + ['Date']]

        # Update boolean conditions
        for i, label in enumerate(boolean_columns):
            if label in conditions_df.columns:
                fig_conditions.data[i].x = dates
                fig_conditions.data[i].y = conditions_df[label].tolist()

        # Update numeric conditions
        for i, label in enumerate(numeric_columns):
            if i + len(boolean_columns) < len(fig_conditions.data):
                fig_conditions.data[i + len(boolean_columns)].x = dates
                fig_conditions.data[i + len(boolean_columns)].y = conditions_df[label].tolist()
            else:
                fig_conditions.add_trace(go.Scatter(x=dates, y=conditions_df[label].tolist(), mode='lines+markers', name=label), row=2, col=1)

    # Load price data
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        fig_data.data[0].x = df['Date']
        fig_data.data[0].y = df['Predicted']
        fig_data.data[1].x = df['Date']
        fig_data.data[1].y = df['Actual']

    # Update layouts
    if 'dates' in locals():
        fig_conditions.update_layout(xaxis_range=[min(dates), max(dates)])
    if 'df' in locals():
        fig_data.update_layout(xaxis_range=[min(df['Date']), max(df['Date'])])

    # Write figures to HTML for viewing
    pio.write_html(fig_conditions, file='fig_conditions.html', auto_open=True)
    pio.write_html(fig_data, file='fig_data.html', auto_open=True)

def update_data_plot(date, predicted_price, actual_price):
    """Update the data plot with new data points."""
    global fig_data
    
    # Convert existing x and y data to lists
    x_pred = list(fig_data.data[0].x)
    y_pred = list(fig_data.data[0].y)
    x_actual = list(fig_data.data[1].x)
    y_actual = list(fig_data.data[1].y)
    
    # Append new data
    x_pred.append(date)
    y_pred.append(predicted_price)
    x_actual.append(date)
    y_actual.append(actual_price)
    
    # Update the traces with the new data
    fig_data.data[0].x = x_pred
    fig_data.data[0].y = y_pred
    fig_data.data[1].x = x_actual
    fig_data.data[1].y = y_actual
    
    # Update the layout
    fig_data.update_layout(
        xaxis=dict(range=[min(x_pred), max(x_pred)]),
        yaxis=dict(range=[
            min(min(y_pred), min(y_actual)),
            max(max(y_pred), max(y_actual))
        ])
    )
    
    pio.write_html(fig_data, file='fig_data.html', auto_open=True)

def update_conditions_plot(condition_values):
    """Update the conditions plot with new data points."""
    global fig_conditions
    
    if not isinstance(condition_values, dict):
        print("Error: condition_values must be a dictionary")
        return
    
    date = condition_values.get('Date')
    if not date:
        print("Error: 'Date' key not found in condition_values")
        return
    
    boolean_columns = ['Volume', 'EMA', 'RSI', 'MACD', 'Bollinger', 'ADX', 'Stochastic K', 'AI', 'Risk', 'Profit', 'Price Jump', 'Golden Cross']
    numeric_columns = [col for col in condition_values.keys() if col not in boolean_columns + ['Date']]
    
    # Update boolean conditions
    for i, label in enumerate(boolean_columns):
        if label in condition_values:
            x = list(fig_conditions.data[i].x)
            y = list(fig_conditions.data[i].y)
            x.append(date)
            y.append(condition_values[label])
            fig_conditions.data[i].x = x
            fig_conditions.data[i].y = y
    
    # Update numeric conditions
    for label in numeric_columns:
        trace = next((trace for trace in fig_conditions.data if trace.name == label), None)
        if trace:
            x = list(trace.x)
            y = list(trace.y)
            x.append(date)
            y.append(condition_values[label])
            trace.x = x
            trace.y = y
        else:
            fig_conditions.add_trace(go.Scatter(x=[date], y=[condition_values[label]], mode='lines+markers', name=label), row=2, col=1)
    
    # Update x-axis range
    all_dates = [trace.x[-1] for trace in fig_conditions.data if len(trace.x) > 0]
    fig_conditions.update_layout(xaxis_range=[min(all_dates), max(all_dates)])
    
    pio.write_html(fig_conditions, file='fig_conditions.html', auto_open=True)

def save_conditions_to_file(condition_values):
    """Save condition values to the CONDITIONS_FILE as CSV."""
    if os.path.exists(CONDITIONS_FILE):
        df = pd.read_csv(CONDITIONS_FILE)
        new_row = pd.DataFrame([condition_values])
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = pd.DataFrame([condition_values])

    # Ensure 'Date' is the last column
    df = df[[col for col in df.columns if col != 'Date'] + ['Date']]

    df.to_csv(CONDITIONS_FILE, index=False)