import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# Load the dataset
file_path = "weather.csv"  # Update with the correct path
df = pd.read_csv(file_path)

# Convert Date/Time to datetime and sort
df["Date/Time (LST)"] = pd.to_datetime(df["Date/Time (LST)"])
df = df.sort_values(by="Date/Time (LST)").reset_index(drop=True)

# Normalize numerical features
num_features = [
    "Temp (°C)", "Dew Point Temp (°C)", "Rel Hum (%)", "Wind Dir (10s deg)",
    "Wind Spd (km/h)", "Visibility (km)", "Stn Press (kPa)", "Hmdx", "Wind Chill", "temp change"
]
scaler = MinMaxScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# Create sequences
SEQ_LENGTH = min(24, len(df) - 1)  # Use 6 past hours to predict the next hour
feature_columns = df.columns.difference(["Date/Time (LST)", "temp change"]).tolist()

seq_data = df[feature_columns].to_numpy()
target_data = df["temp change"].to_numpy()

# Generate rolling sequences
X = np.lib.stride_tricks.sliding_window_view(seq_data, (SEQ_LENGTH, seq_data.shape[1])).squeeze(axis=1)
y = target_data[SEQ_LENGTH:]

# Time-Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
splits = [(train_idx, val_idx) for train_idx, val_idx in tscv.split(X)]

# Define RNN model function (Replaced GRU with SimpleRNN)
def build_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(64, input_shape=input_shape, activation='tanh'),  # Replaced GRU with SimpleRNN
        Dropout(0.5),
        Dense(1)  # Regression output
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# Adjusted splits to avoid out-of-bounds errors
adjusted_splits = [(train_idx[train_idx < len(y)], val_idx[val_idx < len(y)]) for train_idx, val_idx in splits]

# Train and evaluate using time-series cross-validation
history_list = []
for fold, (train_idx, val_idx) in enumerate(adjusted_splits):
    if len(train_idx) == 0 or len(val_idx) == 0:
        print(f"Skipping fold {fold + 1}: index out of bound")
        continue
    else:
        print(f"Training on fold {fold + 1}...")

    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Build and train model
    model = build_rnn_model((SEQ_LENGTH, X.shape[2]))
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=10, batch_size=32, verbose=1)
    
    # Save training history
    history_list.append(history)

# Plot Training Loss
plt.figure(figsize=(10, 5))
for i, history in enumerate(history_list):
    plt.plot(history.history["loss"], label=f"Train Loss Fold {i+1}")
    plt.plot(history.history["val_loss"], label=f"Val Loss Fold {i+1}")

plt.title("Training & Validation Loss Across Folds")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()

