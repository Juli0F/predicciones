from matplotlib import lines
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataframe = pd.read_csv("ventas2.csv");

target = "monto"

independientes = dataframe.drop(columns=target).columns

modelo = LinearRegression();
modelo.fit(X=dataframe[independientes],y=dataframe[target])

dataframe["predicted"] = modelo.predict(dataframe[independientes])
prediccion_test = dataframe[["monto","prediccion"]].head(50)#50 es un numero arbitrario

predicted = modelo.predict([[41,1,1,1]])
print("Posibilidades de Comprar :",predicted);


predicted.plot(kind='bar',figsize=(20,10))
#plt.grid
plt.grid(None)
plt.show()