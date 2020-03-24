
# -- ------------------------------------------------------------------------------------ -- #
# -- Proyecto: Repaso de python 3 y analisis de precios OHLC                              -- #
# -- Codigo: principal.py - script principal de proyecto                                  -- #
# -- Rep: https://github.com/ITESOIF/MyST/tree/master/Notas_Python/Notas_RepasoPython     -- #
# -- Autor: Francisco ME                                                                  -- #
# -- ------------------------------------------------------------------------------------ -- #

# -- ------------------------------------------------------------- Importar con funciones -- #

import funciones as fn                              # Para procesamiento de datos
import visualizaciones as vs                        # Para visualizacion de datos





#%%
df_data = fn.f_leer_archivo(param_archivo='archivo_tradeview_1.xlsx')
