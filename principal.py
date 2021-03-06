
# -- ------------------------------------------------------------------------------------ -- #
# -- Proyecto: Repaso de python 3 y analisis de precios OHLC                              -- #
# -- Codigo: principal.py - script principal de proyecto                                  -- #
# -- Rep: https://github.com/ITESOIF/MyST/tree/master/Notas_Python/Notas_RepasoPython     -- #
# -- Autor: Francisco ME                                                                  -- #
# -- ------------------------------------------------------------------------------------ -- #

# -- ------------------------------------------------------------- Importar con funciones -- #

import funciones as fn                              # Para procesamiento de datos
import visualizaciones as vs                        # Para visualizacion de datos
import pandas as pd
import numpy as np







#%%
#param_archivo='archivo_tradeview_12.csv'
#df_data = fn.f_leer_archivo(param_archivo='archivo_tradeview_12.csv')
param_archivo='tocsv.csv'
df_data = fn.f_leer_archivo(param_archivo='tocsv.csv')

#%%

pip_sizee = fn.f_pip_size(param_ins='btcusd')

#%%

df_data = fn.f_columnas_tiempos(datos = df_data)
#df_temp = pd.DataFrame(temp)
#df_dataf = np.concatenate((df_data, df_temp), axis = 1)
#df_dataf = [df_data[:,:], df_temp[:,:]]

#%%

df_data = fn.f_columnas_pips(datos = df_data)

#%%

f_estadisticas_b = fn.f_estadisticas_ba(datos = df_data)

#%%

df_1_ranking = fn.f_rank(datos = df_data)

#%%

df_data = fn.capital_acm(datos = df_data)

#%%

#df_profit_acm_d = fn.f_profit_diario(datos = df_data)

profit_diario_acum = fn.f_profit_diario(datos = df_data)
#%%

estadisticas_mad = fn.f_estadisticas_mad(datos = df_data)

#%%sesgos cognitivos




#%%

#Graphs1 = fn.graph1(datos = df_1_ranking)
Graphs1 = fn.graph1(datos = df_1_ranking)
#%%
Graphs2 = fn.graph2(input1 = df_data)#, input2 = estadisticas_mad)
#Graphs3 = fn.graph1(datos = df_1_ranking)

#%% graph pytplot
GP1 = fn.gp1(datos = df_1_ranking)

#%%
GP2 = fn.gp2(datos = df_data)



