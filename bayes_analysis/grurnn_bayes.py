import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
import csv
import os
import time

# Display current working directory to confirm where files will be saved
print(f"Files will be saved in: {os.getcwd()}")

# Load data
df = pd.read_csv("weather.csv")
df["Date/Time (LST)"] = pd.to_datetime(df["Date/Time (LST)"])
df = df.sort_values(by="Date/Time (LST)").reset_index(drop=True)

# Set the log filename for GRU model
log_filename = "bayes_gru.csv"
print(f"Logging results to: {os.path.join(os.getcwd(), log_filename)}")

# Create log file and write header if it doesn't exist
if not os.path.exists(log_filename):
    with open(log_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "units", "activation", "epochs", "dropout_rate", "learning_rate", "val_loss", "timestamp"])
        f.flush()  # Force immediate write to disk
    print(f"Created new log file: {log_filename}")
else:
    print(f"Appending to existing log file: {log_filename}")

# Normalize data
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

print(f"Data prepared: X shape = {X.shape}, y shape = {y.shape}")

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=3)
splits = [(train_idx, val_idx) for train_idx, val_idx in tscv.split(X)]
print(f"Created {len(splits)} cross-validation splits")

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
    print(f"\n--- Trial #{trial_counter[0]} (GRU Model) ---")
    print("Params:", params)
    
    # Verify log file is accessible before starting the trial
    try:
        with open(log_filename, mode="a", newline="") as f:
            pass
    except Exception as e:
        print(f"WARNING: Could not access log file: {str(e)}")
    
    fold_losses = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"Training fold {fold_idx+1}/{len(splits)}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # GRU Model
        model = Sequential([
            Input(shape=(SEQ_LENGTH, X.shape[2])),
            GRU(int(params['units']), activation=params['activation']),
            Dropout(params['dropout_rate']),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')
        
        # Train model
        model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val),
            epochs=params['epochs'], 
            batch_size=32, 
            verbose=1
        )
        
        # Evaluate model
        loss = model.evaluate(X_val, y_val, verbose=1)
        fold_losses.append(loss)
        print(f"Fold {fold_idx+1} validation loss: {loss:.6f}")

    avg_loss = np.mean(fold_losses)
    print(f"Average validation loss: {avg_loss:.6f}")

    # Log this trial with timestamp and flush immediately to ensure real-time logging
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                trial_counter[0],
                int(params['units']),
                params['activation'],
                params['epochs'],
                params['dropout_rate'],
                params['learning_rate'],
                avg_loss,
                current_time
            ])
            f.flush()  # Force immediate write to disk
        print(f"Successfully logged trial #{trial_counter[0]} to {log_filename}")
    except Exception as e:
        print(f"ERROR logging to file: {str(e)}")

    return avg_loss

# Run optimization with increased verbosity
print("Starting Bayesian optimization for GRU model...")
res = gp_minimize(
    objective, 
    space, 
    n_calls=20, 
    random_state=42,
    verbose=True  # Show progress during optimization
)

# Best parameters
print("\n" + "="*50)
print("GRU OPTIMIZATION COMPLETE")
print("="*50)
print(f"Best validation score: {res.fun:.6f}")
print("Best hyperparameters:")
for name, val in zip([dim.name for dim in space], res.x):
    print(f"  {name}: {val}")

# Save final results to a separate summary file
summary_filename = "gru_optimization_summary.csv"
with open(summary_filename, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["parameter", "value"])
    writer.writerow(["best_val_loss", res.fun])
    for name, val in zip([dim.name for dim in space], res.x):
        writer.writerow([name, val])
print(f"Summary saved to {summary_filename}")
