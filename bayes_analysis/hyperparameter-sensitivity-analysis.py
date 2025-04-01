import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
import os
import time
import csv

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Function to load best hyperparameters from summary files
def load_best_params(model_type):
    summary_file = f"{model_type.lower()}_optimization_summary.csv"
    if not os.path.exists(summary_file):
        print(f"Warning: {summary_file} not found. Using default parameters.")
        return {
            'units': 64,
            'activation': 'tanh',
            'epochs': 10,
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        }
    
    params = {}
    with open(summary_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                param_name, param_value = row[0], row[1]
                if param_name != 'best_val_loss':
                    # Convert values to appropriate types
                    if param_name in ['units', 'epochs']:
                        params[param_name] = int(float(param_value))
                    elif param_name in ['dropout_rate', 'learning_rate']:
                        params[param_name] = float(param_value)
                    else:
                        params[param_name] = param_value
    
    print(f"Loaded best parameters for {model_type}: {params}")
    return params

# Function to load and prepare data
def prepare_data(filename="weather.csv"):
    print("Loading and preparing data...")
    df = pd.read_csv(filename)
    df["Date/Time (LST)"] = pd.to_datetime(df["Date/Time (LST)"])
    df = df.sort_values(by="Date/Time (LST)").reset_index(drop=True)
    
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
    
    print(f"Data prepared: X shape = {X.shape}, y shape = {y.shape}")
    
    # Create a validation split for consistent evaluation
    tscv = TimeSeriesSplit(n_splits=3)
    train_idx, val_idx = list(tscv.split(X))[-1]  # Use the last split
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    return X_train, X_val, y_train, y_val, SEQ_LENGTH, X.shape[2]

# Function to create and train model
def train_and_evaluate(model_type, params, X_train, X_val, y_train, y_val, input_shape):
    print(f"Training {model_type} model with params: {params}")
    
    # Build model based on type
    model = Sequential([
        Input(shape=input_shape),
        GRU(params['units'], activation=params['activation']) if model_type == 'GRU' else
        LSTM(params['units'], activation=params['activation']),
        Dropout(params['dropout_rate']),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']), 
        loss='mse'
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params['epochs'],
        batch_size=32,
        verbose=0
    )
    
    # Evaluate model
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    return val_loss

# Function to perform sensitivity analysis
def sensitivity_analysis(model_type, base_params, X_train, X_val, y_train, y_val, input_shape):
    results = {
        'units': [],
        'learning_rate': [],
        'dropout_rate': []
    }
    
    # Vary units
    units_values = [8, 16, 32, 64, 96, 128, 160, 192, 224, 256]
    for units in units_values:
        params = base_params.copy()
        params['units'] = units
        val_loss = train_and_evaluate(model_type, params, X_train, X_val, y_train, y_val, input_shape)
        results['units'].append((units, val_loss))
        print(f"{model_type} - Units: {units}, Val Loss: {val_loss:.6f}")
    
    # Vary learning rate
    lr_values = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]
    for lr in lr_values:
        params = base_params.copy()
        params['learning_rate'] = lr
        val_loss = train_and_evaluate(model_type, params, X_train, X_val, y_train, y_val, input_shape)
        results['learning_rate'].append((lr, val_loss))
        print(f"{model_type} - Learning Rate: {lr}, Val Loss: {val_loss:.6f}")
    
    # Vary dropout rate
    dropout_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for dropout in dropout_values:
        params = base_params.copy()
        params['dropout_rate'] = dropout
        val_loss = train_and_evaluate(model_type, params, X_train, X_val, y_train, y_val, input_shape)
        results['dropout_rate'].append((dropout, val_loss))
        print(f"{model_type} - Dropout Rate: {dropout}, Val Loss: {val_loss:.6f}")
    
    return results

# Function to plot sensitivity analysis results
def plot_results(results_gru, results_lstm):
    plt.figure(figsize=(18, 14))
    
    # Plot units sensitivity
    plt.subplot(3, 1, 1)
    plt.plot(*zip(*results_gru['units']), 'o-', label='GRU')
    plt.plot(*zip(*results_lstm['units']), 's-', label='LSTM')
    plt.xscale('log')
    plt.xlabel('Units (log scale)')
    plt.ylabel('Validation Loss')
    plt.title('Effect of Units on Model Performance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot learning rate sensitivity
    plt.subplot(3, 1, 2)
    plt.plot(*zip(*results_gru['learning_rate']), 'o-', label='GRU')
    plt.plot(*zip(*results_lstm['learning_rate']), 's-', label='LSTM')
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Validation Loss')
    plt.title('Effect of Learning Rate on Model Performance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot dropout rate sensitivity
    plt.subplot(3, 1, 3)
    plt.plot(*zip(*results_gru['dropout_rate']), 'o-', label='GRU')
    plt.plot(*zip(*results_lstm['dropout_rate']), 's-', label='LSTM')
    plt.xlabel('Dropout Rate')
    plt.ylabel('Validation Loss')
    plt.title('Effect of Dropout Rate on Model Performance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hyperparameter_sensitivity.png', dpi=300)
    plt.close()
    
    # Also save individual plots for each model type
    for model_type, results in [('GRU', results_gru), ('LSTM', results_lstm)]:
        plt.figure(figsize=(16, 12))
        
        # Units
        plt.subplot(3, 1, 1)
        plt.plot(*zip(*results['units']), 'o-', color='blue')
        plt.xscale('log')
        plt.xlabel('Units (log scale)')
        plt.ylabel('Validation Loss')
        plt.title(f'{model_type}: Effect of Units on Model Performance')
        plt.grid(True, alpha=0.3)
        
        # Learning rate
        plt.subplot(3, 1, 2)
        plt.plot(*zip(*results['learning_rate']), 'o-', color='green')
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Validation Loss')
        plt.title(f'{model_type}: Effect of Learning Rate on Model Performance')
        plt.grid(True, alpha=0.3)
        
        # Dropout rate
        plt.subplot(3, 1, 3)
        plt.plot(*zip(*results['dropout_rate']), 'o-', color='red')
        plt.xlabel('Dropout Rate')
        plt.ylabel('Validation Loss')
        plt.title(f'{model_type}: Effect of Dropout Rate on Model Performance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{model_type.lower()}_sensitivity.png', dpi=300)
        plt.close()
    
    print("Plots saved: hyperparameter_sensitivity.png, gru_sensitivity.png, lstm_sensitivity.png")

# Save results to CSV
def save_results(results_gru, results_lstm):
    # Save GRU results
    with open('gru_sensitivity_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['parameter_type', 'parameter_value', 'validation_loss'])
        for param_type, values in results_gru.items():
            for value, loss in values:
                writer.writerow([param_type, value, loss])
    
    # Save LSTM results
    with open('lstm_sensitivity_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['parameter_type', 'parameter_value', 'validation_loss'])
        for param_type, values in results_lstm.items():
            for value, loss in values:
                writer.writerow([param_type, value, loss])
    
    print("Results saved to gru_sensitivity_results.csv and lstm_sensitivity_results.csv")

# Main execution
def main():
    start_time = time.time()
    print("Starting hyperparameter sensitivity analysis...")

    # Load best parameters for each model type
    gru_params = load_best_params('GRU')
    lstm_params = load_best_params('LSTM')
    
    # Prepare data
    X_train, X_val, y_train, y_val, seq_length, n_features = prepare_data()
    input_shape = (seq_length, n_features)
    
    # Run sensitivity analysis for GRU
    print("\n==== GRU Sensitivity Analysis ====")
    results_gru = sensitivity_analysis('GRU', gru_params, X_train, X_val, y_train, y_val, input_shape)
    
    # Run sensitivity analysis for LSTM
    print("\n==== LSTM Sensitivity Analysis ====")
    results_lstm = sensitivity_analysis('LSTM', lstm_params, X_train, X_val, y_train, y_val, input_shape)
    
    # Plot and save results
    plot_results(results_gru, results_lstm)
    save_results(results_gru, results_lstm)
    
    elapsed_time = time.time() - start_time
    print(f"\nSensitivity analysis completed in {elapsed_time/60:.2f} minutes")

if __name__ == "__main__":
    main()
