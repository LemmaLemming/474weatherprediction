import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

# Load CSV
csv_data = "weather.csv"
df = pd.read_csv(csv_data)

# Parse Date/Time and sort
df['Date/Time (LST)'] = pd.to_datetime(df['Date/Time (LST)'])
df = df.sort_values(by='Date/Time (LST)')

# Create persistence model predictions (next value is current value)
df['Persistence Prediction'] = df['temp change'].shift(1)

# Drop NA values caused by shifting
df.dropna(subset=['Persistence Prediction'], inplace=True)

# Calculate RMSE
y_true = df['temp change']
y_pred = df['Persistence Prediction']
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"Root Mean Squared Error: {rmse:.4f}\n")

# Plot true vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(df['Date/Time (LST)'], y_true, label='Actual', alpha=0.7)
plt.plot(df['Date/Time (LST)'], y_pred, label='Persistence Model', alpha=0.7)
plt.title('Persistence Model Prediction vs Actual Temp Change')
plt.xlabel('Time')
plt.ylabel('Temp Change (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and show the plot
plot_path = "persistence_model_plot.png"
plt.savefig(plot_path)
plt.show()
