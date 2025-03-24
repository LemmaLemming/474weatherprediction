import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

df = pd.read_csv("weather.csv")
df["Date/Time (LST)"] = pd.to_datetime(df["Date/Time (LST)"])

# Create sequences
SEQ_LENGTH = min(6, len(df) -1)  # Use 6 past hours to predict the next hour
feature_columns = df.columns.difference(["Date/Time (LST)", "temp change"]).tolist()

seq_data = df[feature_columns].to_numpy()
target_data = df["temp change"].to_numpy()

# Generate rolling sequences
X = np.lib.stride_tricks.sliding_window_view(seq_data, (SEQ_LENGTH, seq_data.shape[1])).squeeze(axis=1)
y = target_data[SEQ_LENGTH:]

# Time-Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
splits = [(train_idx, val_idx) for train_idx, val_idx in tscv.split(X)]

def build_rnn_model(input_shape):
    model = Sequential([
        GRU(64, input_shape=input_shape),
        Dropout(0.2),
        Dense(1)  
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


adjusted_splits = [(train_idx[train_idx < len(y)], val_idx[val_idx < len(y)]) for train_idx, val_idx in splits]
# Train and evaluate using time-series cross-validation
history_list = []
for fold, (train_idx, val_idx) in enumerate(adjusted_splits):
    if len(train_idx) == 0 or len(val_idx) == 0:
        print(f"Skipping fold {fold +1}: index out of bound")
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
