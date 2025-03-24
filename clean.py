import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

data = pd.read_csv("./2013-2025weatherData.csv")

data["Temp (°C)"]= pd.to_numeric(data["Temp (°C)"], errors='coerce')
data = data.dropna(subset = ["Temp (°C)"])
data = data.dropna(axis=1,how="all")

data["Time (LST)"]  = pd.to_datetime(data['Time (LST)'], format='%H:%M').dt.hour
data.drop(["Longitude (x)", 
           "Latitude (y)",
           "Station Name",
           "Climate ID",
           "Dew Point Temp Flag" ,
           "Rel Hum Flag",
           "Precip. Amount Flag",
           "Wind Dir Flag",
           "Wind Spd Flag",
           "Visibility Flag",
           "Stn Press Flag"]
          ,inplace = True,axis=1)

data = data.sort_values("Date/Time (LST)")
data["temp change"] =round( data["Temp (°C)"].diff(),2)

if "Weather" in data.columns:
    label_encoder = LabelEncoder()
    data["Weather"] = label_encoder.fit_transform(data["Weather"])

data = data.fillna(0)

#Change these features to be [0,1]
num_features = [
    "Day", "Month", "Year", "Temp (°C)", "Dew Point Temp (°C)", "Rel Hum (%)", "Wind Dir (10s deg)",
    "Wind Spd (km/h)", "Visibility (km)", "Stn Press (kPa)", "Hmdx", "Wind Chill", "temp change"
]
scaler = MinMaxScaler()
data[num_features] = scaler.fit_transform(data[num_features])

print(data)
data.to_csv("weather.csv",index=False)

