#Reactividad alpha
from turtle import color
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import random

def select_color(col):
    color_col = []
    for color in col:
        r = random.random()
        b = random.random()
        g = random.random()
        c = (r, g, b)
        c=list(c)
        color_col.append(c[:3])
    return color_col

CE=pd.read_feather(r"E:\Academico\Universidad\Posgrado\Tesis\Datos\BASESDEDATOS\BIOMARCADORES_BIDS\derivatives\longitudinal_data_powers_long_CE_components.feather") 
OE=pd.read_excel(r"E:\Academico\Universidad\Posgrado\Tesis\Paquetes\sovaConectivity_Reactivity\longitudinal_data_powers_long_OE_components.xlsx")

col_CE = CE.columns.values
col_OE = OE.columns.values  
datos_CE = CE[CE['Bands'].str.contains('alpha')]
datos_OE = OE[OE['Bands'].str.contains('alpha')]
datos_CE.reset_index(drop=True, inplace=True)
datos_OE.reset_index(drop=True, inplace=True)
for sbj in range(len(datos_OE['Powers'])):
    color_col = select_color(range(len(datos_OE['Powers'])))
    try:
        reactividad = (datos_CE['Powers'][sbj]) - (datos_OE['Powers'][sbj])/datos_CE['Powers'][sbj]
        #print(reactividad)

        plt.scatter(sbj,datos_CE['Powers'][sbj],marker="v",label='CE',color=color_col[sbj])
        plt.scatter(sbj,datos_OE['Powers'][sbj],marker="°",label='OE',color=color_col[sbj])
        plt.ylabel('Amplitud ')
        plt.xlabel('Frecuencia')
        plt.title('Señal')
        plt.grid()
        plt.legend()
        plt.show()

    except:
        pass




