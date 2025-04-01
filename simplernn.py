import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
import tensorflow as tf
import csv

# Load dataset
df = pd.read_csv("weather.csv")
df['Date/Time (LST)'] = pd.to_datetime(df['Date/Time (LST)'])
df = df.sort_values(by='Date/Time (LST)')

# Select features and target
features = ['Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Dir (10s deg)',
            'Wind Spd (km/h)', 'Visibility (km)', 'Stn Press (kPa)', 'Hmdx',
            'Wind Chill']  # Removed 'Weather' as it's categorical and needs different handling
target = 'temp change'

df = df[features + [target]].dropna()

# Scale the features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features].values)
labels = df[target].values

# Define a function to create sequences for RNN
def create_sequences(data, labels, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(labels[i + window_size])
    return np.array(X), np.array(y)

# Search space for hyperparameters
search_space = [
    Integer(12, 48, name='window_size'),
    Integer(1, 3, name='num_layers'),
    Integer(16, 128, name='units'),
    Real(1e-4, 1e-2, "log-uniform", name='learning_rate'),
    Real(0.0, 0.5, name='dropout'),
    Real(0.0, 1e-3, name='l2_reg'),
    Integer(16, 64, name='batch_size')
]

# CSV setup for logging results
csv_file = "rnn_tuning_results.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['window_size', 'num_layers', 'units', 'learning_rate',
                     'dropout', 'l2_reg', 'batch_size', 'avg_val_mae'])

@use_named_args(search_space)
def objective(**params):
    window_size = params['window_size']
    X, y = create_sequences(scaled_data, labels, window_size)

    tscv = TimeSeriesSplit(n_splits=5)  # Adjust number of splits as needed
    val_maes = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = Sequential()
        model.add(Input(shape=(window_size, len(features))))
        for i in range(params['num_layers']):
            model.add(SimpleRNN(
                int(params['units']),
                activation='tanh',
                return_sequences=(i < params['num_layers'] - 1),
                kernel_regularizer=l2(params['l2_reg'])
            ))
            model.add(Dropout(params['dropout']))
        model.add(Dense(1))

        optimizer = Adam(learning_rate=params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=10,  # Adjust number of epochs as needed
                            batch_size=params['batch_size'],
                            verbose=0)

        _, val_mae = model.evaluate(X_val, y_val, verbose=0)
        val_maes.append(val_mae)

    avg_val_mae = np.mean(val_maes)

    # Log results to CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            params['window_size'], params['num_layers'], params['units'],
            params['learning_rate'], params['dropout'], params['l2_reg'],
            params['batch_size'], avg_val_mae
        ])

    print(f"Params: {params}, Avg MAE: {avg_val_mae:.4f}")
    return avg_val_mae

# Run the optimization
n_calls = 15  # Adjust the number of optimization calls
results = gp_minimize(
    func=objective,
    dimensions=search_space,
    n_calls=n_calls,
    random_state=42
)

# Best result
best_params = dict(zip(
    ['window_size', 'num_layers', 'units', 'learning_rate',
     'dropout', 'l2_reg', 'batch_size'],
    results.x
))

print("\nBest Hyperparameters:")
print(best_params)
