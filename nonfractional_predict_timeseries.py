import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

import FractionalOptimizers as fr          # your custom fractional optimizers
import fractal_activation_functions as fractal  # your fractal activation functions

##############################################################################
# 1) Hyperparameters and Setup
##############################################################################
FRAC_DERIV = 1.25
#ACTIVATION_FN = fractal.decaying_cosine_function_tf  # fractal activation
#ACTIVATION_FN = fractal.modified_weierstrass_function_relu  # fractal activation
#ACTIVATION_FN = fractal.modified_weierstrass_function_tanh  # fractal activation
#ACTIVATION_FN = fractal.new_fav_fractal_function  # fractal activation
#ACTIVATION_FN = fractal.weierstrass_mandelbrot_function_xsinsquared
#ACTIVATION_FN = fractal.weierstrass_mandelbrot_function_tanhpsin
#ACTIVATION_FN = fractal.modulated_blancmange_curve #best so far
# ACTIVATION_FN = fractal.decaying_cosine_function_tf  # Fractal activation
# ACTIVATION_FN = fractal.decaying_sine_function_tf    # Example: Another fractal activation
ACTIVATION_FN = tf.keras.activations.relu            # Standard ReLU
# ACTIVATION_FN = tf.keras.activations.sigmoid         # Sigmoid activation
#ACTIVATION_FN = tf.keras.activations.tanh            # Hyperbolic tangent
# ACTIVATION_FN = tf.keras.activations.swish           # Swish activation
# ACTIVATION_FN = tf.keras.activations.softplus        # Softplus activation
#OPTIMIZER_FN = fr.FRMSprop(vderiv=FRAC_DERIV, learning_rate=0.01)  # fractional RMSprop works wiht standarad activation function
#OPTIMIZER_FN = fr.FRMSprop(vderiv=FRAC_DERIV, learning_rate=0.1)  # fractional RMSprop
# Uncomment one of these optimizers as needed:
# OPTIMIZER_FN = fr.FRMSprop                           # Fractional RMSprop
# OPTIMIZER_FN = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)  # SGD with momentum
#OPTIMIZER_FN = tf.keras.optimizers.Adam(learning_rate=0.01)              # Adam optimizer
# OPTIMIZER_FN = tf.keras.optimizers.Adagrad(learning_rate=0.01)            # Adagrad optimizer
OPTIMIZER_FN = tf.keras.optimizers.RMSprop(learning_rate=0.01)           # RMSprop optimizer
#OPTIMIZER_FN = tf.keras.optimizers.Adamax(learning_rate=0.002)
EPOCHS = 50 #100 works great
BATCH_SIZE = 8 #batch 8 wokrs for standard activation function
WINDOW_SIZE = 12
TEST_RATIO = 0.33
CSV_FILENAME = "./time_series_data/international-airline-passengers.csv"

##############################################################################
# 2) Data Loading
##############################################################################
def load_airline_passengers_data(csv_file):
    """
    Loads the International Airline Passengers CSV from Kaggle,
    dropping or filling any NaN rows in 'Passengers'.
    """
    df = pd.read_csv(csv_file)
    if "Passengers" not in df.columns:
        raise ValueError("CSV must contain a 'Passengers' column!")

    # Drop rows where 'Passengers' is NaN (or fill them).
    df = df.dropna(subset=["Passengers"])
    # Alternatively, could fill with forward fill:
    # df["Passengers"].fillna(method="ffill", inplace=True)

    data = df["Passengers"].values.astype(float)
    return data

##############################################################################
# 3) Detrending + Scaling
##############################################################################
def detrend_and_scale(data, feature_range=(0.1, 0.9)):
    """
    Fits a linear trend y = a + b*t over the entire series, subtracts it,
    and then scales the residual to [0.1..0.9].
    """
    N = len(data)
    t = np.arange(N).reshape(-1, 1)

    # Fit linear regression for trend
    linreg = LinearRegression()
    linreg.fit(t, data)
    trend_vals = linreg.predict(t)

    # Subtract trend
    data_detrended = data - trend_vals

    # Scale residual to [0.1, 0.9]
    scaler = MinMaxScaler(feature_range=feature_range)
    data_scaled = scaler.fit_transform(data_detrended.reshape(-1, 1)).flatten()
    return data_scaled, trend_vals, scaler

def inverse_detrend_and_scale(value_scaled, index, trend_vals, scaler):
    """
    Inverse transforms a single scaled/detrended prediction:
      - first inverse-scaling
      - then re-add the linear trend at 'index'
    """
    val_detrended = scaler.inverse_transform([[value_scaled]])[0, 0]
    return val_detrended + trend_vals[index]

##############################################################################
# 4) Create Sliding Window (One-Step-Ahead)
##############################################################################
def create_sliding_window(series, window_size=10):
    """
    For each i, X[i] = series[i : i+window_size], y[i] = series[i+window_size].
    """
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

##############################################################################
# 5) Build the Model
##############################################################################
def build_model(input_dim):
    """
    LSTM-based model using a fractal activation + fractional RMSProp.
    """
    model = tf.keras.Sequential()
    # LSTM expects input shape: (timesteps, features)
    model.add(
        tf.keras.layers.LSTM(
            48, activation=ACTIVATION_FN, input_shape=(input_dim, 1)
        )
    )
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    frac_optimizer = OPTIMIZER_FN
    model.compile(optimizer=frac_optimizer, loss='mean_squared_error')
    return model

##############################################################################
# 6) Main
##############################################################################
def main():
    # --------------------------------------------------------------------------
    # (A) Load raw data
    # --------------------------------------------------------------------------
    raw_data = load_airline_passengers_data(CSV_FILENAME)
    N = len(raw_data)
    print(f"Loaded data shape: {raw_data.shape}. Example: {raw_data[:5]}...")

    # --------------------------------------------------------------------------
    # (B) Detrend and scale
    # --------------------------------------------------------------------------
    data_scaled, trend_vals, scaler = detrend_and_scale(
        raw_data, feature_range=(0.1, 0.9)
    )

    # --------------------------------------------------------------------------
    # (C) Build sliding windows (X_all, y_all)
    # --------------------------------------------------------------------------
    X_all, y_all = create_sliding_window(data_scaled, window_size=WINDOW_SIZE)

    # --------------------------------------------------------------------------
    # (D) Split into train/test
    # --------------------------------------------------------------------------
    train_size = int(len(X_all) * (1 - TEST_RATIO))
    X_train, X_test = X_all[:train_size], X_all[train_size:]
    y_train, y_test = y_all[:train_size], y_all[train_size:]
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Reshape for LSTM: (samples, timesteps, features=1)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # (for reference) indexes in the *original* time series:
    #   For X_train[i], y_train[i] => time index = i + WINDOW_SIZE
    #   For X_test[i],  y_test[i]  => time index = (train_size + i) + WINDOW_SIZE
    train_start_idx_in_original = WINDOW_SIZE
    test_start_idx_in_original = train_size + WINDOW_SIZE

    # --------------------------------------------------------------------------
    # (E) Build & Train model
    # --------------------------------------------------------------------------
    model = build_model(input_dim=WINDOW_SIZE)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1
    )

    # ==========================================================================
    # (F) PREDICTIONS ON TRAINING SET
    # ==========================================================================
    #
    # (F1) Direct single-step predictions on each training sample
    # --------------------------------------------------------------------------
    y_train_pred_scaled = model.predict(X_train).squeeze()  # shape=(train_size,)
    preds_train_direct = []
    for i, val_scaled in enumerate(y_train_pred_scaled):
        idx_in_original = train_start_idx_in_original + i
        val_orig = inverse_detrend_and_scale(
            val_scaled, idx_in_original, trend_vals, scaler
        )
        preds_train_direct.append(val_orig)
    preds_train_direct = np.array(preds_train_direct)

    # (F2) Autoregressive (rolling) approach on training
    # --------------------------------------------------------------------------
    # Start with the first training window (X_train[0])
    predictions_scaled_ar_train = []
    last_known_window_train = X_train[0].copy()  # shape=(WINDOW_SIZE,1)

    # We roll through the entire training set
    for i in range(len(X_train)):
        pred_scaled = model.predict(last_known_window_train.reshape(1, WINDOW_SIZE, 1))[0, 0]
        predictions_scaled_ar_train.append(pred_scaled)

        # Slide the window, appending the new pred
        last_known_window_train = np.roll(last_known_window_train, -1, axis=0)
        last_known_window_train[-1] = pred_scaled

    # Convert scaled preds to original domain
    preds_train_ar = []
    for i, val_scaled in enumerate(predictions_scaled_ar_train):
        idx_in_original = train_start_idx_in_original + i
        val_orig = inverse_detrend_and_scale(
            val_scaled, idx_in_original, trend_vals, scaler
        )
        preds_train_ar.append(val_orig)
    preds_train_ar = np.array(preds_train_ar)

    # ==========================================================================
    # (G) PREDICTIONS ON TEST SET
    # ==========================================================================
    #
    # (G1) Direct single-step test predictions
    # --------------------------------------------------------------------------
    y_test_pred_scaled = model.predict(X_test).squeeze()  # shape=(test_size,)
    preds_test_direct = []
    for i, val_scaled in enumerate(y_test_pred_scaled):
        idx_in_original = test_start_idx_in_original + i
        val_orig = inverse_detrend_and_scale(
            val_scaled, idx_in_original, trend_vals, scaler
        )
        preds_test_direct.append(val_orig)
    preds_test_direct = np.array(preds_test_direct)

    # (G2) Autoregressive (rolling) approach on test
    # --------------------------------------------------------------------------
    predictions_scaled_ar_test = []
    last_known_window_test = X_test[0].copy()  # shape=(WINDOW_SIZE,1)

    for i in range(len(X_test)):
        pred_scaled = model.predict(last_known_window_test.reshape(1, WINDOW_SIZE, 1))[0, 0]
        predictions_scaled_ar_test.append(pred_scaled)

        # Slide the window, appending the new pred
        last_known_window_test = np.roll(last_known_window_test, -1, axis=0)
        last_known_window_test[-1] = pred_scaled

    # Convert scaled preds to original domain
    preds_test_ar = []
    for i, val_scaled in enumerate(predictions_scaled_ar_test):
        idx_in_original = test_start_idx_in_original + i
        val_orig = inverse_detrend_and_scale(
            val_scaled, idx_in_original, trend_vals, scaler
        )
        preds_test_ar.append(val_orig)
    preds_test_ar = np.array(preds_test_ar)

    # ==========================================================================
    # (H) Convert ground truth y_train, y_test to original
    # ==========================================================================
    y_train_original = []
    for i, val_scaled in enumerate(y_train):
        idx_in_original = train_start_idx_in_original + i
        val_orig = inverse_detrend_and_scale(
            val_scaled, idx_in_original, trend_vals, scaler
        )
        y_train_original.append(val_orig)
    y_train_original = np.array(y_train_original)

    y_test_original = []
    for i, val_scaled in enumerate(y_test):
        idx_in_original = test_start_idx_in_original + i
        val_orig = inverse_detrend_and_scale(
            val_scaled, idx_in_original, trend_vals, scaler
        )
        y_test_original.append(val_orig)
    y_test_original = np.array(y_test_original)

    # ==========================================================================
    # (I) Plot all
    # ==========================================================================
    t_full = np.arange(N)

    # Training portion indexes: from i=0..(train_size-1), time = i + WINDOW_SIZE
    t_train = np.arange(train_start_idx_in_original, train_start_idx_in_original + len(y_train))
    # Test portion indexes
    t_test = np.arange(test_start_idx_in_original, test_start_idx_in_original + len(y_test))

    plt.figure(figsize=(12,7), dpi=120)

    # 1) Original full series
    plt.plot(
        t_full, raw_data,
        label="Original Series",
        color='black', linewidth=2.0
    )

    # 2) Train ground truth portion
    plt.plot(
        t_train, y_train_original,
        label="Train Ground Truth",
        color='blue', linewidth=2.0
    )

    # 3) Test ground truth portion
    plt.plot(
        t_test, y_test_original,
        label="Test Ground Truth",
        color='green', linewidth=2.0
    )

    # -- Training Predictions --
    plt.plot(
        t_train, preds_train_direct,
        label="Train Direct (1-step) Pred",
        color='magenta', linestyle=':', linewidth=2.5
    )
    plt.plot(
        t_train, preds_train_ar,
        label="Train AR (rolling) Pred",
        color='cyan', linestyle='--', linewidth=2.5
    )

    # -- Testing Predictions --
    plt.plot(
        t_test, preds_test_direct,
        label="Test Direct (1-step) Pred",
        color='orange', linestyle=':', linewidth=3.0
    )
    plt.plot(
        t_test, preds_test_ar,
        label="Test AR (rolling) Pred",
        color='red', linestyle='--', linewidth=2.5
    )

    plt.title("Airline Passengers - LSTM with Train & Test Predictions")
    plt.xlabel("Time Index (Monthly)")
    plt.ylabel("Passengers Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # (J) Plot training history
    plt.figure(figsize=(8,4), dpi=120)
    plt.plot(history.history["loss"], label="Train Loss", color='blue')
    plt.plot(history.history["val_loss"], label="Val Loss", color='orange')
    plt.title("Training/Validation Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
