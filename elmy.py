import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt

X_train = pd.read_csv('/home/mr-burns/Bureau/code/Data_science/Elmy/X_train', )


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

print(df)
# print(df.head())
print(df.isnull().sum())

spot_id = pd.read_csv('/home/mr-burns/Bureau/code/Data_science/Elmy/y_train')
print(spot_id)

x = df.iloc[:, -1] 
y = df.iloc[:, -1]

plt.plot(x, y)
plt.show()