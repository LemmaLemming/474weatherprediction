import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler



# builds a RNN model (2 LSTM layers and 1 output layer)
def build_rnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, activation='tanh', return_sequences=True),
        LSTM(50, activation='tanh'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# creates sequences of data and target variables of a given sequence length
def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)


# load and prepare data
df = pd.read_csv("weather.csv")
df["Date/Time (LST)"] = pd.to_datetime(df["Date/Time (LST)"])

# define sequence length (last 24 hours to predict the next hour)
SEQ_LENGTH = min(24, len(df) - 1)

# create sequences of data and target variables
feature_columns = df.columns.difference(["Date/Time (LST)", "temp change"]).tolist()
seq_data = df[feature_columns].to_numpy()
target_data = df["temp change"].to_numpy()

# create a MinMaxScaler for feature and target data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

# normalize the feature and target data
seq_data_scaled = scaler_X.fit_transform(seq_data)
target_data_scaled = scaler_y.fit_transform(target_data.reshape(-1, 1)).flatten()

# create sequences with normalized data
X, y = create_sequences(seq_data_scaled, target_data_scaled, SEQ_LENGTH)

# create TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
splits = [(train_idx, val_idx) for train_idx, val_idx in tscv.split(X)]




# train and evaluate for 10 epochs on each fold
history_list = []
for fold, (train_idx, val_idx) in enumerate(splits):
    print(f"Training on fold {fold + 1}...")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = build_rnn_model((SEQ_LENGTH, X.shape[-1]))
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=10, batch_size=32, verbose=1)
    
    history_list.append(history)




# plot training and validation loss across all folds and epochs
plt.figure(figsize=(10, 5))
for i, history in enumerate(history_list):
    plt.plot(history.history["loss"], label=f"Train Loss Fold {i+1}")
    plt.plot(history.history["val_loss"], label=f"Val Loss Fold {i+1}")

plt.title("Training & Validation Loss Across Epochs")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()

