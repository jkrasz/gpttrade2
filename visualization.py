import pandas as pd
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from config import CONDITIONS_FILE, DATA_FILE, symbol
import json

fig_conditions = None
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

def update_data_plot(date, predicted_price, actual_price):
    """Update the data plot with new data points."""
    global fig_data
    
    fig_data.add_trace(go.Scatter(x=[date], y=[predicted_price], mode='lines', name='Predicted Price'))
    fig_data.add_trace(go.Scatter(x=[date], y=[actual_price], mode='lines', name='Actual Price'))
    
    fig_data.update_layout(
        xaxis=dict(range=[min(fig_data.data[0].x + fig_data.data[1].x), max(fig_data.data[0].x + fig_data.data[1].x)]),
        yaxis=dict(range=[min(fig_data.data[0].y + fig_data.data[1].y), max(fig_data.data[0].y + fig_data.data[1].y)])
    )
    
    fig_data.show()

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
            fig_conditions.add_trace(go.Scatter(x=[date], y=[condition_values[label]], mode='lines+markers', name=label), row=1, col=1)
    
    # Update numeric conditions
    for i, label in enumerate(numeric_columns):
        fig_conditions.add_trace(go.Scatter(x=[date], y=[condition_values[label]], mode='lines+markers', name=label), row=2, col=1)
    
    fig_conditions.show()
