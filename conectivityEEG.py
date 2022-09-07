from conectivity import sl_connectivity 
import time 
import pandas as pd
import feather
import pandas as pd
from datasets import BIOMARCADORES_CE as DATA 

#df = pd.read_feather(r'sovaConectivity_Reactivity\Matrix_LS.feather')
THE_DATASETS=[DATA]
for dataset in THE_DATASETS:
    start = time.perf_counter()
    sl_subject,sl_columns=sl_connectivity(dataset,fast_mode=False)
    sl_df = pd.DataFrame(sl_subject)
    sl_df = sl_df.T
    sl_df.columns = sl_columns
    sl_df.to_feather('sovaConectivity_Reactivity\Matrix_LS_CE.feather')
    final = time.perf_counter()
    print('TIME PREPROCESSING:::::::::::::::::::', final-start)
    start = time.perf_counter()
    final = time.perf_counter()
    print('TIME CREATE FEATHERS:::::::::::::::::::', final-start)