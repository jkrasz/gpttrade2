import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer,Dense,LayerNormalization,GRU, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional, BatchNormalization, Activation
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import RMSprop

class TransformerEncoderBlock(Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout=0):
        super(TransformerEncoderBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(dense_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_transformer_model(input_shape, units=64, dropout_rate=0.2, attention_units=32, l1_reg=0.01, l2_reg=0.01):
    embed_dim = input_shape[1]  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = tf.keras.Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = TransformerEncoderBlock(embed_dim, ff_dim, num_heads, dropout=dropout_rate)(x)
    x = Flatten()(x)
    x = Dense(units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(attention_units, activation='relu')(x)
    outputs = Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=RMSprop(), loss='mean_squared_error')
    return model

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

def build_gru_model(input_shape, units=64, dropout_rate=0.2, attention_units=32, l1_reg=0.01, l2_reg=0.01):
    model = Sequential([
        Conv1D(filters=64, kernel_size=5, padding='same', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), input_shape=input_shape),
        Activation('relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Bidirectional(GRU(units, return_sequences=True)),
        Dropout(dropout_rate),
        Bidirectional(GRU(units, return_sequences=True)),
        Dropout(dropout_rate),
        Dense(attention_units, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=RMSprop(), loss='mean_squared_error')
    return model



def build_cnn_model(input_shape, units=64, dropout_rate=0.2, attention_units=32, l1_reg=0.01, l2_reg=0.01):
    model = Sequential([
        Conv1D(filters=64, kernel_size=5, padding='same', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), input_shape=input_shape),
        Activation('relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=5, padding='same', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        Activation('relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(units, activation='relu'),
        Dropout(dropout_rate),
        Dense(attention_units, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=RMSprop(), loss='mean_squared_error')
    return model

def predict_price(model, gru_model, cnn_model, transformer_model, data, scaler, sequence_length=60):
    """
    Make a prediction for the next closing price.
    """
    # Ensure the last sequence is a 2D array as expected by scaler
    last_sequence = data[-sequence_length:].reshape(-1, 1)  # Reshaping to 2D array
    last_sequence_scaled = scaler.transform(last_sequence)
    last_sequence_scaled = np.reshape(last_sequence_scaled, (1, sequence_length, 1))
    
    # Predict with each model
    predicted_scaled_lstm = model.predict(last_sequence_scaled)
    predicted_scaled_gru = gru_model.predict(last_sequence_scaled)
    predicted_scaled_cnn = cnn_model.predict(last_sequence_scaled)
    predicted_scaled_transformer = transformer_model.predict(last_sequence_scaled)
    
    # Average predictions
    predicted_scaled = (predicted_scaled_lstm + predicted_scaled_gru + predicted_scaled_cnn + predicted_scaled_transformer) / 4
    
    # Inverse transform to get the predicted price
    predicted_price = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))[0, 0]
    
    return predicted_price