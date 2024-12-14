import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt

X_train = pd.read_csv('/home/mr-burns/Bureau/Elmy/X_train')



df = X_train.rename(columns={
    'DELIVERY_START': "date et heure de livraison de l'électricité",
    'load_forecast': "prévision de consommation totale en France",
    'coal_power_available': "production totale des centrales à charbon", 
    'gas_power_available': "production totale des centrales à gaz",
    'nucelear_power_available': "production totale des centrales nucléaire", 
    'wind_power_forecasts_average': "moyenne prévisions de production éolienne", 
    'solar_power_forecasts_average': "moyenne prévisions de production solaire", 
    'wind_power_forecasts_std': "écart type de production éolienne", 
    'solar_power_forecasts_std': "écart type de production solaire", 
    'predicted_spot_price': "prévision du prix SPOT"
    }
               )

df.dropna()

print(df.head())

print(df)
# print(df.head())
print(df.isnull().sum())

spot_id = pd.read_csv('/home/mr-burns/Bureau/Elmy/y_train')
print(spot_id)

import pandas as pd
from sklearn.linear_model import LinearRegression


# Conversion des dates en timestamp
df[:, -1] = pd.to_datetime(df['date'])
spot_id[:, -1] = df['date'].astype(int) / 10**9  # ou une autre transformation si nécessaire


# Créer et entraîner le modèle
model = LinearRegression()
model.fit(df, spot_id)
