import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional, BatchNormalization, Activation
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import RMSprop


def build_lstm_model(input_shape, units=64, dropout_rate=0.2, attention_units=32, l1_reg=0.01, l2_reg=0.01):
    model = Sequential([
        Conv1D(filters=64, kernel_size=5, padding='same', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), input_shape=input_shape),
        Activation('relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(units, return_sequences=True)),
        Dropout(dropout_rate),
        Bidirectional(LSTM(units, return_sequences=True)),
        Dropout(dropout_rate),
        Dense(attention_units, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=RMSprop(), loss='mean_squared_error')
    return model

def predict_price(model, data, scaler, sequence_length=60):
    """
    Make a prediction for the next closing price.
    """
    # Ensure the last sequence is a 2D array as expected by scaler
    last_sequence = data[-sequence_length:].reshape(-1, 1)  # Reshaping to 2D array
    last_sequence_scaled = scaler.transform(last_sequence)
    last_sequence_scaled = np.reshape(last_sequence_scaled, (1, sequence_length, 1))
    predicted_scaled = model.predict(last_sequence_scaled)
    predicted_price = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))[0, 0]
    
    return predicted_price