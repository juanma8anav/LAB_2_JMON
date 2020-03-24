
# -- ------------------------------------------------------------------------------------ -- #
# -- Proyecto: Describir brevemente el proyecto en general                                -- #
# -- Codigo: RepasoPython.py - describir brevemente el codigo                             -- #
# -- Repositorio: https://github.com/                                                     -- #
# -- Autor: Nombre de autor                                                               -- #
# -- ------------------------------------------------------------------------------------ -- #

# import numpy as np                                      # funciones numericas
import pandas as pd                                       # dataframes y utilidades
from datetime import timedelta                            # diferencia entre datos tipo tiempo


# -- --------------------------------------------------------- FUNCION: Descargar precios -- #
# -- Descargar precios historicos con OANDA
def f_leer_archivo(param_archivo):

    """
    Funcion para leer archivo en formato xlsx.
    :param param_archivo: Cadena de texto con el nombre del archivo
    :return: dataframe de datos importados de archivo excel
    Debugging
    ---------
    param_archivo = 'archivo_tradeview_1.xlsx'
    """
 # Leer archivo de datos y guardarlo en Data Frame
    df_data = pd.read_csv(r'C:/Users/juanm/Documents/Iteso/Sem10/Trading/labWork/' + param_archivo)#, sheet_name='archivo_tradeview_1')
   # Convertir a minusculas el nombre de las columnas
    df_data.columns = [list(df_data.columns)[i].lower() for i in range(0, len(df_data.columns))]
    # Asegurar que ciertas columnas son tipo numerico
    numcols =  ['s/l', 't/p', 'commission', 'openprice', 'closeprice', 'profit', 'size', 'swap', 'taxes']
    df_data[numcols] = df_data[numcols].apply(pd.to_numeric)    
    return df_data
    
#%%

def f_pip_size(param_ins):

    """"
    Parameters
    ----------
    param_ins : str : nombre de instrumento 
    Returns
    -------
    pips_inst : 
    Debugging
    ---------
    """""
    #param_ins =  param_archivo
    # encontrar y eliminar un guion bajo
    inst = param_ins.replace('_', '')
    inst = param_ins.replace('-2', '')    
    # transformar a minusculas
    inst = inst.lower()
    #lista de pips por instrumento
     pip_inst = {'usdjpy': 100, 'gbpjpy': 100, 'eurjpy': 100,
                'eurusd': 10000, 'gbpusd': 10000, 'usdcad': 10000, 'usdmxn': 10000,
                'audusd': 10000, 'eurgbp': 10000,
                'xauusd': 10, 'btcusd': 1}
    #{'xauusd': 10, 'eurusd': 10000, 'xaueur': 10,'bcousd':1000,'conrusd':10000 ,'mbtcusd':1000,'wtiusd':1000, 'spx500usd':10}
    
    return pips_inst[param_ins]
