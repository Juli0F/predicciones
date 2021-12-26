import numpy as np
import pandas as pd
from pandas.core.algorithms import isin, mode
from scipy.sparse import data
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
from sklearn.linear_model import LinearRegression

#def read_csv(path):

#lee el csv
def read_data(file_path):
    df = pd.read_csv(file_path)
    return df;

def select_number_data(data_frame):

    if(isinstance(data_frame,pd.DataFrame)):
        datos_numericos = data_frame.select_dtypes(np.number)
        datos_numericos = data_frame.select_dtypes(np.number).fillna(0)
        return datos_numericos;
    
#estable la variable dependiente
# y devuelve las variables dependientes    
def set_target(data_frame,target):

    if(isinstance(data_frame,pd.DataFrame)):
        independientes = data_frame.drop(columns=target).columns
        return independientes;

def set_model(data_frame,independientes,datos_numericos,target):

    modelo = LinearRegression()
    modelo.fit(X=datos_numericos[independientes], y=datos_numericos[target])
    data_frame["predicted"] = modelo.predict(datos_numericos[independientes])
    return data_frame;



data_frame = read_data("./movies2.csv");
datos_numericos = select_number_data(data_frame=data_frame);
independientes = set_target(datos_numericos,"ventas")
predicted = set_model(data_frame,independientes,datos_numericos,"ventas")
print(predicted[["ventas","predicted"]].head())

print("---"*20)
data_frame = read_data("./ventas2.csv");
datos_numericos = select_number_data(data_frame=data_frame);
independientes = set_target(datos_numericos,"monto")
predicted = set_model(data_frame,independientes,datos_numericos,"monto")
print(predicted[["monto","predicted"]].head())
        
        