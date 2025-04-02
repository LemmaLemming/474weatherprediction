import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

def build_rnn_model():
    model = Sequential([
        Input(shape=(SEQ_LENGTH, X.shape[-1])),
        LSTM(50, activation='tanh', return_sequences=True, recurrent_dropout=0.01),
        LSTM(50, activation='tanh', recurrent_dropout=0.01),
        Dense(1)
   ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)



df = pd.read_csv("weather.csv")
df["Date/Time (LST)"] = pd.to_datetime(df["Date/Time (LST)"])
SEQ_LENGTH = min(24, len(df) - 1)

feature_columns = df.columns.difference(["Date/Time (LST)", "temp change"]).tolist()
seq_data = df[feature_columns].to_numpy()
target_data = df["temp change"].to_numpy()

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))


seq_data_scaled = scaler_X.fit_transform(seq_data)
target_data_scaled = scaler_y.fit_transform(target_data.reshape(-1, 1)).flatten()
X, y = create_sequences(seq_data_scaled, target_data_scaled, SEQ_LENGTH)


model = build_rnn_model();
history = model.fit(X, y, epochs=10, batch_size=32, verbose=1)

split_indx = int(len(X) * 0.8)
X_test =  X[split_indx:]
y_test =  y[split_indx:]

y_predicted = model.predict(X_test)

y_pred = scaler_y.inverse_transform(y_predicted.reshape(-1, 1)).flatten()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
print(f"RMSE: {rmse:.3f}")

plt.figure(figsize=(14, 6))
plt.plot(y_test_actual, label='Actual Change in Temperature', linewidth=2)
plt.plot(y_pred, label='Predicted Change in Temperature', linestyle='--')
plt.fill_between(range(len(y_pred)),
                 y_pred - rmse,
                 y_pred + rmse,
                 color='orange',
                 alpha=0.2,
                 label=f'± RMSE ({rmse:.2f}°C)')
plt.title("Actual vs Predicted Temperature Change (Next Hour)")
plt.xlabel("Time (1 step = 1 hour)")
plt.ylabel("Temperature Change (°C)")
plt.xlim(0, 500) 
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()










