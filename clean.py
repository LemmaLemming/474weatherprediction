import pandas as pd

data = pd.read_csv("./2013-2025weatherData.csv")

data["Temp (°C)"]= pd.to_numeric(data["Temp (°C)"], errors='coerce')
data = data.dropna(subset = ["Temp (°C)"])
data = data.dropna(axis=1,how="all")
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
data = data.fillna(0)
print(data)
data.to_csv("weather.csv",index=False)

