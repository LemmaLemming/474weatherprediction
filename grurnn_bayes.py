import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
import csv

# Load data
df = pd.read_csv("weather.csv")
df["Date/Time (LST)"] = pd.to_datetime(df["Date/Time (LST)"])
df = df.sort_values(by="Date/Time (LST)").reset_index(drop=True)
log_file = open("bayes_gru_log.csv", mode="w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Iteration", "units", "activation", "epochs", "dropout_rate", "learning_rate", "MSE"])

trial_log = []

# Normalize
num_features = [
    "Temp (°C)", "Dew Point Temp (°C)", "Rel Hum (%)", "Wind Dir (10s deg)",
    "Wind Spd (km/h)", "Visibility (km)", "Stn Press (kPa)", "Hmdx", "Wind Chill", "temp change"
]
scaler = MinMaxScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# Sequence generation
SEQ_LENGTH = min(24, len(df) - 1)
feature_columns = df.columns.difference(["Date/Time (LST)", "temp change"]).tolist()
seq_data = df[feature_columns].to_numpy()
target_data = df["temp change"].to_numpy()

windows = sliding_window_view(seq_data, window_shape=(SEQ_LENGTH, seq_data.shape[1]))
X = windows.squeeze(axis=1)
y = target_data[SEQ_LENGTH - 1:]


tscv = TimeSeriesSplit(n_splits=3)
splits = [(train_idx, val_idx) for train_idx, val_idx in tscv.split(X)]

# Hyperparameter space
space = [
    Integer(16, 128, name='units'),
    Categorical(['tanh', 'relu', 'sigmoid'], name='activation'),
    Integer(3, 20, name='epochs'),
    Real(0.1, 0.5, name='dropout_rate'),
    Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate')
]

trial_counter = [0]  # To track number of trials

@use_named_args(space)
def objective(**params):
    trial_counter[0] += 1
    print(f"\n--- Trial #{trial_counter[0]} ---")
    print("Params:", params)

    fold_losses = []
    for train_idx, val_idx in splits:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = Sequential([
            Input(shape=(SEQ_LENGTH, X.shape[2])),
            GRU(int(params['units']), activation=params['activation']),
            Dropout(params['dropout_rate']),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=params['epochs'], batch_size=32, verbose=0)
        loss = model.evaluate(X_val, y_val, verbose=1)
        fold_losses.append(loss)

    avg_loss = np.mean(fold_losses)

    # Log the trial
    trial_log.append({
        'trial': trial_counter[0],
        'units': int(params['units']),
        'activation': params['activation'],
        'epochs': params['epochs'],
        'dropout_rate': params['dropout_rate'],
        'learning_rate': params['learning_rate'],
        'val_loss': avg_loss
    })

    return avg_loss

res = gp_minimize(objective, space, n_calls=20, random_state=42)

# Save trial log to CSV
log_file = "bayes_gru_trials_log.csv"
with open(log_file, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=trial_log[0].keys())
    writer.writeheader()
    writer.writerows(trial_log)

print(f"\nLog saved to {log_file}")


# Best params
print("Best score: {:.5f}".format(res.fun))
print("Best hyperparameters:")
for name, val in zip([dim.name for dim in space], res.x):
    print(f"  {name}: {val}")
