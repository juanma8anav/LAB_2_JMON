# -- ------------------------------------------------------------------------------------ -- #
# -- proyecto: Microestructura y Sistemas de Trading - Laboratorio 2 - Behavioral Finance
# -- archivo: funciones.py - procesamiento de datos
# -- mantiene: mauanaya
# -- repositorio: https://github.com/mauanaya/LAB_2_VMAA
# -- ------------------------------------------------------------------------------------ -- #

import pandas as pd


# -- -------------------------------------------------------------------- FUNCION: leer archivo -- #
def f_leer_archivo(param_archivo):
    """"
    Parameters
    ----------
    param_archivos : str : nombre de archivo a leer

    Returns
    -------
    df_data : pd.DataFrame : con informacion contenida en archivo leido

    Debugging
    ---------
    param_archivo =  'archivo_tradeview_1.xlsx'

    """
    # Leer archivo de datos y guardarlo en Data Frame
    df_data = pd.read_excel(r'C:\Users\Usuario\Documents\Sem9\Trading\LAB_2_JMON\' + param_archivo, sheet_name='Hoja1')

    # Convertir a minusculas el nombre de las columnas
    df_data.columns = [list(df_data.columns)[i].lower() for i in range(0, len(df_data.columns))]

    # Asegurar que ciertas columnas son tipo numerico
    numcols = ['s/l', 't/p', 'commission', 'openprice', 'closeprice', 'profit', 'size', 'swap', 'taxes', 'order']
    df_data[numcols] = df_data[numcols].apply(pd.to_numeric)

    return df_data

