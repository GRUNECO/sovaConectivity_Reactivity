#Reactividad alpha
import numpy as np
import pandas as pd 
import collections
import scipy.io
from tokenize import group
import pingouin as pg
from scipy import stats

CE=pd.read_feather(r"E:\Academico\Universidad\Posgrado\Tesis\Reactividad\longitudinal_data_powers_long_CE_components.feather") 
OE=pd.read_feather(r"E:\Academico\Universidad\Posgrado\Tesis\Reactividad\longitudinal_data_powers_long_OE_components.feather")
datos_CE = CE[CE['Bands'].str.contains('alpha')]
datos_OE = OE[OE['Bands'].str.contains('alpha')]
reactividad = (datos_CE - datos_OE)/datos_CE