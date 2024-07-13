import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LayerNormalization, GRU, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional, BatchNormalization, Activation, MultiHeadAttention, Layer, Input, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

class TransformerEncoderBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_advanced_lstm_model(input_shape, units=128, dropout_rate=0.3, l1_reg=0.01, l2_reg=0.01):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(units, return_sequences=True, kernel_regularizer=l1_l2(l1_reg, l2_reg)))(inputs)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(units, return_sequences=True, kernel_regularizer=l1_l2(l1_reg, l2_reg)))(x)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(units, return_sequences=False, kernel_regularizer=l1_l2(l1_reg, l2_reg)))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
    return model

def build_advanced_gru_model(input_shape, units=128, dropout_rate=0.3, l1_reg=0.01, l2_reg=0.01):
    inputs = Input(shape=input_shape)
    x = Bidirectional(GRU(units, return_sequences=True, kernel_regularizer=l1_l2(l1_reg, l2_reg)))(inputs)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(GRU(units, return_sequences=True, kernel_regularizer=l1_l2(l1_reg, l2_reg)))(x)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(GRU(units, return_sequences=False, kernel_regularizer=l1_l2(l1_reg, l2_reg)))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
    return model

def build_advanced_transformer_model(input_shape, num_transformer_blocks=3, embed_dim=128, num_heads=8, ff_dim=256, dropout_rate=0.2, l1_reg=0.01, l2_reg=0.01):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=embed_dim, kernel_size=1, padding='same', activation='relu', kernel_regularizer=l1_l2(l1_reg, l2_reg))(inputs)
    for _ in range(num_transformer_blocks):
        x = TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
    return model

def build_advanced_cnn_model(input_shape, filters=128, kernel_size=3, dropout_rate=0.3, l1_reg=0.01, l2_reg=0.01):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', kernel_regularizer=l1_l2(l1_reg, l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=filters*2, kernel_size=kernel_size, padding='same', kernel_regularizer=l1_l2(l1_reg, l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
    return model

def train_models(X, y, input_shape, epochs=100, batch_size=32):
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001)
    ]

    lstm_model = build_advanced_lstm_model(input_shape)
    gru_model = build_advanced_gru_model(input_shape)
    transformer_model = build_advanced_transformer_model(input_shape)
    cnn_model = build_advanced_cnn_model(input_shape)

    lstm_model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=callbacks, verbose=0)
    gru_model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=callbacks, verbose=0)
    transformer_model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=callbacks, verbose=0)
    cnn_model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=callbacks, verbose=0)

    X_2d = X.reshape(X.shape[0], -1)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    svr_model = SVR(kernel='rbf')

    rf_model.fit(X_2d, y)
    gb_model.fit(X_2d, y)
    xgb_model.fit(X_2d, y)
    svr_model.fit(X_2d, y)

    return lstm_model, gru_model, transformer_model, cnn_model, rf_model, gb_model, xgb_model, svr_model

def predict_price(models, data, scaler, sequence_length=60):
    lstm_model, gru_model, transformer_model, cnn_model, rf_model, gb_model, xgb_model, svr_model = models
    last_sequence = data[-sequence_length:].reshape(-1, 1)
    last_sequence_scaled = scaler.transform(last_sequence)
    last_sequence_scaled_3d = np.reshape(last_sequence_scaled, (1, sequence_length, 1))
    last_sequence_scaled_2d = last_sequence_scaled.reshape(1, -1)

    predicted_scaled_lstm = lstm_model.predict(last_sequence_scaled_3d)
    predicted_scaled_gru = gru_model.predict(last_sequence_scaled_3d)
    predicted_scaled_transformer = transformer_model.predict(last_sequence_scaled_3d)
    predicted_scaled_cnn = cnn_model.predict(last_sequence_scaled_3d)
    predicted_scaled_rf = rf_model.predict(last_sequence_scaled_2d)
    predicted_scaled_gb = gb_model.predict(last_sequence_scaled_2d)
    predicted_scaled_xgb = xgb_model.predict(last_sequence_scaled_2d)
    predicted_scaled_svr = svr_model.predict(last_sequence_scaled_2d)

    ensemble_prediction = np.mean([
        predicted_scaled_lstm,
        predicted_scaled_gru,
        predicted_scaled_transformer,
        predicted_scaled_cnn,
        predicted_scaled_rf,
        predicted_scaled_gb,
        predicted_scaled_xgb,
        predicted_scaled_svr
    ])

    predicted_price = scaler.inverse_transform(ensemble_prediction.reshape(-1, 1))[0, 0]

    return predicted_price