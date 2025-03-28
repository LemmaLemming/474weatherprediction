import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

# builds a RNN model (2 LSTM layers and 1 output layer, and adds respective hyperparameters)
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

# plots the training and validation loss across all folds and epochs for each tuned model
def plot_loss(history_list, hyperparameter_name, hyperparameters):
    plt.figure(figsize=(10, 5))
    plt.title("Training & Validation Loss Across Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")

    for i, history in enumerate(history_list):
        plt.plot(history.history["loss"], label=f"Train Loss {hyperparameter_name} = {hyperparameters[hyperparameter_name]}")
        plt.plot(history.history["val_loss"], label=f"Val Loss {hyperparameter_name} = {hyperparameters[hyperparameter_name]}")

    plt.legend()
    plt.show()

    
def investigate_hyperparameter(hyperparameter_name, hyperparameters, X, y, SEQ_LENGTH, splits):
    # store history for each hyperparameter value
    all_histories = {}

    for value in hyperparameters[hyperparameter_name]:
        print(f"Training on {hyperparameter_name} = {value}")

        history_list = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"Fold {fold + 1}...")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # choosing hyperparameters 
            if hyperparameter_name == "units":
                    model = Sequential([
                        Input(shape=(SEQ_LENGTH, X.shape[-1])),
                        LSTM(value, activation='tanh', return_sequences=True),
                        LSTM(value, activation='tanh'),
                        Dense(1)
                    ])
            elif hyperparameter_name == "activation":
                model = Sequential([
                        Input(shape=(SEQ_LENGTH, X.shape[-1])),
                        LSTM(50, activation=value, return_sequences=True),
                        LSTM(50, activation=value),
                        Dense(1)
                    ])
            elif hyperparameter_name == "dropout":
                model = Sequential([
                        Input(shape=(SEQ_LENGTH, X.shape[-1])),
                        LSTM(50, activation='tanh', return_sequences=True, dropout=value),
                        LSTM(50, activation='tanh', dropout=value),
                        Dense(1)
                    ])
                
            elif hyperparameter_name == "recurrent_dropout":
                model = Sequential([
                        Input(shape=(SEQ_LENGTH, X.shape[-1])),
                        LSTM(50, activation='tanh', return_sequences=True, recurrent_dropout=value),
                        LSTM(50, activation='tanh', recurrent_dropout=value),
                        Dense(1)
                    ])

            model.compile(optimizer='adam', loss='mean_squared_error')

            history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                epochs=10, batch_size=32, verbose=1)
            
            history_list.append(history)

        # Store histories for this hyperparameter value
        all_histories[value] = history_list

    # plot training & validation loss for each hyperparameter setting
    plt.figure(figsize=(12, 6))
    plt.title(f"Effect of {hyperparameter_name} on Model Performance")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error (MSE)")

    for value, histories in all_histories.items():
        avg_train_loss = np.mean([h.history["loss"] for h in histories], axis=0)
        avg_val_loss = np.mean([h.history["val_loss"] for h in histories], axis=0)

        plt.plot(avg_train_loss, label=f"Train {hyperparameter_name}={value}")
        plt.plot(avg_val_loss, linestyle="dashed", label=f"Val {hyperparameter_name}={value}")

    plt.legend()
    plt.show()



# load and prepare data
df = pd.read_csv("weather.csv")
df["Date/Time (LST)"] = pd.to_datetime(df["Date/Time (LST)"])


# define sequence length (last 24 hours to predict the next hour)
SEQ_LENGTH = min(24, len(df) - 1)

# define hyperparameters to tune
hyperparameters_list = {
    'activation': ['tanh', 'relu'],
    'units': [5, 10, 20, 50, 70, 100],
    'dropout': [0.1, 0.2, 0.3],
    'recurrent_dropout': [0.1, 0.2, 0.3],
    }


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

print("investigateing hyperparameters...")
# investigate_hyperparameter("activation", hyperparameters_list, X, y, SEQ_LENGTH, splits)
# investigate_hyperparameter("units", hyperparameters_list, X, y, SEQ_LENGTH, splits)
investigate_hyperparameter("dropout", hyperparameters_list, X, y, SEQ_LENGTH, splits)
# investigate_hyperparameter("recurrent_dropout", hyperparameters_list, X, y, SEQ_LENGTH, splits)



# train and evaluate for 10 epochs on each fold
# history_list = []
# for fold, (train_idx, val_idx) in enumerate(splits):
#     print(f"Training on fold {fold + 1}...")

#     X_train, X_val = X[train_idx], X[val_idx]
#     y_train, y_val = y[train_idx], y[val_idx]

#     model = build_rnn_model((SEQ_LENGTH, X.shape[-1]))
#     history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
#                         epochs=10, batch_size=32, verbose=1)
    
#     history_list.append(history)
    
    
    
# train and evaluate for 10 epochs on each fold with different hyperparameters...






