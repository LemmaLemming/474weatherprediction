import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler


# Load and prepare your data
weatherData = pd.read_csv('weather.csv')
# drop_columns = ['Date/Time (LST)']
# weatherData = weatherData.drop(drop_columns, axis=1)


# Create sequences
SEQ_LENGTH = min(24, len(weatherData) -1)  # Use 24 past hours to predict the next hour
feature_columns = weatherData.columns.difference(["Date/Time (LST)", "temp change"]).tolist()

seq_data = weatherData[feature_columns].to_numpy()
target_data = weatherData["temp change"].to_numpy()

# Generate rolling sequences
X = np.lib.stride_tricks.sliding_window_view(seq_data, (SEQ_LENGTH, seq_data.shape[1])).squeeze(axis=1)
y = target_data[SEQ_LENGTH:]


tscv = TimeSeriesSplit(n_splits=5)


# To store training and validation errors
train_errors = []
val_errors = []

splits = [(train_idx, val_idx) for train_idx, val_idx in tscv.split(X)]

adjusted_splits = [(train_idx[train_idx < len(y)], val_idx[val_idx < len(y)]) for train_idx, val_idx in splits]


for curFold ,train_idx, val_idx in enumerate(adjusted_splits):
    if len(train_idx) == 0 or len(val_idx) == 0:
        print("Skipping fold", curFold, "due to empty training or validation set")
        continue    
    else:
        print("Fold", curFold, "Train size:", len(train_idx), "Validation size:", len(val_idx))
        
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Build LSTM model
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),  
        LSTM(50, activation='tanh', return_sequences=True),
        LSTM(50, activation='tanh'),
        Dense(1)  # Single output for prediction
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model and track the training/validation loss for each epoch
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=0)
    
    # # Store the training and validation error per epoch for each fold
    # train_errors_fold = history.history['loss']  # Training loss per epoch
    # val_errors_fold = history.history['val_loss']  # Validation loss per epoch
    # train_errors.append(train_errors_fold)
    # val_errors.append(val_errors_fold)


# # Split data into features (X) and target (Y)
# X = weatherData.iloc[:, :-1].values  # All columns except last
# Y = weatherData.iloc[:, -1].values   # Last column as target

# # Min-Max scaling for normalization
# scaler_X = MinMaxScaler(feature_range=(0, 1))
# scaler_Y = MinMaxScaler(feature_range=(0, 1))

# X_scaled = scaler_X.fit_transform(X)
# Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1))

# # Reshape X to be 3D as required by LSTM [samples, timesteps, features]
# X_scaled_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))







# Loop over each train/test split for TimeSeriesSplit
for train_index, test_index in tscv.split(X_scaled_reshaped):
    # Split data into training and testing sets based on the indices
    X_train, X_test = X_scaled_reshaped[train_index], X_scaled_reshaped[test_index]
    Y_train, Y_test = Y_scaled[train_index], Y_scaled[test_index]
    
    # Build LSTM model
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),  
        LSTM(50, activation='tanh', return_sequences=True),
        LSTM(50, activation='tanh'),
        Dense(1)  # Single output for prediction
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model and track the training/validation loss for each epoch
    history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test), verbose=0)
    
    # Store the training and validation error per epoch for each fold
    train_errors_fold = history.history['loss']  # Training loss per epoch
    val_errors_fold = history.history['val_loss']  # Validation loss per epoch
    train_errors.append(train_errors_fold)
    val_errors.append(val_errors_fold)
    

    

# Calculate the average training and validation error across all folds
avg_train_error = np.mean(train_errors, axis=0)  # Average training error over folds
avg_val_error = np.mean(val_errors, axis=0)  # Average validation error over folds

# Plot training and validation errors for each epoch
plt.plot(range(1, len(avg_train_error) + 1), avg_train_error, label='Training Error (MSE)', color='blue')
plt.plot(range(1, len(avg_val_error) + 1), avg_val_error, label='Validation Error (MSE)', color='red')
plt.title('Training and Validation Error per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.show()
