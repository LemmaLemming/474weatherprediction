import pandas as pd
data = pd.read_csv("./2013-2025weatherData.csv")
data["Temp (°C)"]= pd.to_numeric(data["Temp (°C)"], errors='coerce')
data = data.dropna(subset = ["Temp (°C)"])
data = data.dropna(axis=1,how="all")
data = data.sort_values("Date/Time (LST)")
data["temp change"] = data["Temp (°C)"].diff()
print(data)
data.to_csv("weather.csv",index=False)

