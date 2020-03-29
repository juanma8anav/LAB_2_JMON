
# -- ------------------------------------------------------------------------------------ -- #
# -- Proyecto: Describir brevemente el proyecto en general                                -- #
# -- Codigo: RepasoPython.py - describir brevemente el codigo                             -- #
# -- Repositorio: https://github.com/                                                     -- #
# -- Autor: Nombre de autor                                                               -- #
# -- ------------------------------------------------------------------------------------ -- #

import numpy as np                                      # funciones numericas
import pandas as pd                                       # dataframes y utilidades
from datetime import timedelta                            # diferencia entre datos tipo tiempo
import datetime as datetime


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
    datcols = ['opentime', 'closetime']
    df_data[datcols] = df_data[datcols].apply(pd.to_datetime)
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
    param_ins =  'btcusd'
    # encontrar y eliminar un guion bajo
    inst = param_ins.replace('_', '')
    inst = param_ins.replace('-2', '')    
    # transformar a minusculas
    inst = inst.lower()
    #lista de pips por instrumento
    pip_inst = pip_inst = {'usdjpy': 100, 'gbpjpy': 100, 'eurjpy': 100, 'cadjpy': 100,
                'chfjpy': 100,'eurusd': 10000, 'gbpusd': 10000, 'usdcad': 10000, 
                'usdmxn': 10000,'audusd': 10000, 'nzdusd': 10000, 'usdchf': 10000, 
                'eurgbp': 10000, 'eurchf': 10000, 'eurnzd': 10000, 'euraud': 10000, 
                'gbpnzd': 10000, 'gbpchf': 10000, 'gbpaud': 10000, 'audnzd': 10000, 
                'nzdcad': 10000,'audcad': 10000, 'xauusd': 10, 'xagusd': 10, 'btcusd': 1}
    #{'xauusd': 10, 'eurusd': 10000, 'xaueur': 10,'bcousd':1000,'conrusd':10000 ,'mbtcusd':1000,'wtiusd':1000, 'spx500usd':10}
    
    return pip_inst[param_ins]

#%%
def f_columnas_tiempos(datos):
    """
    Función que toma el ¨closetime¨ y el ¨opentime¨
    y regresa la diferencia entre esos dos en segundos 
    para saber cuanto tiempo duró abierta cada operación
    -----------------------------------------------------
    df_data : nombre de la base de datos con la que estamos trabajando
    
    """
    
    i = 0
    #temp = []
    for i in range(0, len(datos)):#np.size(datos,2)):
        #datos['tiempo'] = datos.iloc[i,6]-datos.iloc[i,1]
        datos['tiempo'] = pd.to_datetime(datos.iloc[i,6])-pd.to_datetime(datos.iloc[i,1])
        #tiempop.astype('timedelta64[D]')
        #tiempop = datos.iloc[-6] - datos.iloc[2]
        #tiempop = np.timedelta64(datos.iloc[6]) - np.timedelta64(datos.iloc[1])  
        #np.datetime64(df_data.iloc[-6]) - np.datetime64(df_data.iloc[2])
        #tiempops = np.datetime64(tiempop, 's')
        #temp.append(tiempop)
        i = i+1
    #td =   pd.to_timedelta(temp) 
    #td = pd.to_timedelta(['-1 days +02:45:00','1 days +02:45:00','0 days +02:45:00'])
    #df = pd.DataFrame({'td': td})

    #df['td'] = df['td'] - pd.to_timedelta(df['td'].dt.days, unit='d')

    #df = pd.DataFrame(pd.Timestamp(temp))
    #j=0    
    #for j in range(0, 84):
        
    return datos

#%%
def f_columnas_pips(datos):
    datos['pips'] = np.zeros(len(datos))
    i=0
    for i in range(0, len(datos)):
        #if datos.iloc[2, i] == "buy":
        if datos['type'][i] == 'buy':
            datos['pips'][i] = (datos.closeprice[i] - datos.openprice[i])*f_pip_size(param_ins=datos['symbol'][i])
        else:
            datos['pips'][i] = (datos.closeprice[i] - datos.openprice[i])*f_pip_size(param_ins=datos['symbol'][i])*-1
            
    datos['pips_acm'] = np.zeros(len(datos))
    datos['pips_acm'] = datos.pips.cumsum()
    datos['profit_acm'] = np.zeros(len(datos))
    datos['profit_acm'] = datos.profit.cumsum()
            
    return datos

#%%

def f_estadisticas_ba(datos):
    c = 0
    v1 = 0
    v2 = 0
    v3 = 0
    v4 = v5 =v6 =v7=v8=v9=v10=v11=v12=0
    v0 = len(datos)
    for c in range(0,len(datos)):#len(filter(datos['profit'] if x>0 )) #sum(1 for x in datos['profit'] if datos.iloc[i, 'profit'] >0)   # sum(1 for x in my_list if datos.profit[x]>0)
        if datos.profit[c]>0:
            v1 = v1+1
            
        c=+1
     
    prev2 = datos[(datos.type == 'buy') & (datos.profit > 0)]
    v2 = len(prev2)
    #v2 = datos[datos[type == 'buy'] & datos['profit'] > 0].count()
    #sum(1 for v2 in v2 if datos['type'][i] == 'buy' and datos['profit'][i] > 0)
#    c= 0
#    
#    for i in range(0,len(datos['profit'])):  
#        if datos['type'][i] == 'buy' and datos['profit'][i] > 0 :
#            v3 =+1
#    c = 0
#    for c in range(0,len(datos)):
#        if datos['type'][c] == 'buy' and datos['profit'][c] > 0:
#            v3 =+1
#        c =+1

   # sum(1 if datos.iloc[c,2] == 'buy' and datos.iloc[c,-5] > 0 for c in range (0, len(datos)) in v3)
    
    #sum(1 )
        
        
#    c=0        
#    for c in range(0,len(datos['size'])):
#        if datos.iloc[c,2] == 'buy' and datos.iloc[c,-5] > 0:
#            
#            v3 =+ 1
        
                
            

    


#    c=0        
#    for c in range(0,len(datos)):
#        if datos.profit.iloc[c]>0 and datos['type'][c] == 'sell':
#            v4 =+ 1
#            
#        c=+1 
#    
#    c=0 
#    for c in range(0,len(datos)): # v5 = len(datos) - vv2
#        if datos.profit.iloc[c]<0:
#            v5 =+ 1
#            
#        c=+1
#    
#        c=0        
#    for c in range(0,len(datos)):
#        if datos.profit.iloc[c]<0 and datos['type'][c] == 'buy':
#            v6 =+ 1
#            
#        c=+1
#    
#        c=0        
#    for c in range(0,len(datos)):
#        if datos.profit.iloc[c]<0 and datos['type'][c] == 'bsell':
#            v7 =+ 1
#            
#        c=+1
        
        
    v7 = np.median(datos.profit)
    
    v8 = np.median(datos.pips)
    
    v9 =v1/v0
    
    v10 = 1- v9
    
    v11 = v2/v0
    
    v12 =1-v11
    
    _tabla = {'Medida' : ['Ops totales', 'Ganadoras', 'Ganadoras_c', 'Ganadoras_v', 'Perdedoras', 
                            'Perdedoras_c', 'Perdedoras_v', 'Media(Profit)', 'Media(pips)', 'r_efectividad',
                            'r_proporcion', 'r_efectividad_c', 'r_efectividad_v'], 
                            'valor' : [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12], 
                            'descripcion' : ['Operaciones totales', 
                             'Operaciones ganadoras', 'Operaciones ganadoras de compra', 'Operaciones ganadoras de venta',
                             'Operaciones perdedoras', 'Operaciones perdedoras de compra', 'Operaciones perdedoras de venta',
                             'Mediana de prfit de operaciones', 'Mediana pips de operaciones', 'Ganadoras Totales/Operaciones Totales',
                             'Perdedoras Totales/Ganadoras Totales', 'Ganadoras Compras/Operaciones Totales', 'Ganadoras Ventas/ Operaciones Totales']}
 

   
    df_1_tabla = pd.DataFrame(_tabla)
     
    return df_1_tabla
 
    
#%%
    
def f_rank(datos):
    
    #ssymbol = datos['symbol'].name.unique
    
    ssymb = []
    for i in datos['symbol']:
        if i not in ssymb:
            ssymb.append(i)
            
    rnk = np.zeros(len(ssymb))
    
    
    
    
    
    df_1_ranking = pd.DataFrame({'Symbol' : ssymb, 'Rank' : rnk})
    
    
#    datos.sort_values(by = ['symbol'])#, ascending = True)
#    i = 0 
#    j = 0 
#    
#    for i in range(0, len(datos)):
#        for j in range (0, len(df_1_ranking)):
#            if datos['symbol'][i] == df_1_ranking['Symbol'][j]:
#                if datos['profit'][i] > 0:
#                    sum(1 in df_1_ranking['Rank'][j])
#                else:
#                    j =+1
#            i =+1
                        
    
    
    
    
#    for datos.index, datos['symbol'] in datos.iterrows():
#        for df_1_ranking.index, df_1_ranking['Symbol'] in df_1_ranking.iterrows():
#            if ((datos['sysmbol'] == df_1_ranking['Symbol']) and (datos['profit'] > 0)):
#                sum(1 in df_1_ranking['Rank'])
    
#    adid = np.zeros(len(datos)-len(df_1_ranking),2)
#    df_adid = pd.DataFrame(adid)
#    
#    df_1_ranking = pd.concat(df_1_ranking, df_adid)
    
#    x = 0
#    for x in range(0, len(df_data)):
#        if pd.merge(df_1_ranking, df_data, on = (df_1_ranking['Symbol'] == df_data['symbol'])):
#            if df_data['profit'] > 0:
#                sum(1 in df_1_ranking['Rank'][x])
#                x =+1
    
    
    
    #sum(1 for x in df_1_ranking['Rank'] if df_data[df_data['symbol'] ==])
    
    
#    i = 0
#    for i in range(0, len(df_data)):
#        if df_data['profit'].eq(0)
   
        
    
    
#    array = [-37,-36,-19,-99,29,20,3,-7,-64,84,36,62,26,-76,55,-24,84,49,-65,41] 
#    print sum(i for i in array if array.index(i) % 2 == 0)*array[-1] if array != [] else 0
#    i = 0    
#    sum(1 for i in df_1_ranking['Rank'] if datos['symbol'][i] == df_1_ranking['Symbol'] & datos['profit'][i] > 0) 
    
    
#    q = 0
#    x = 0
#    for q in range(0, len(datos)):
#       sum(1 for x in df_1_ranking['Rank'][x] if df_1_ranking['Symbol'] == datos['symbol'] and datos['profit'] >0) / sum(1 if df_1_ranking['Symbol'][x] == datos['symbol'][x])
    
#    i = 0
#    j = 0
#    c = 0
#    t = 0
#
#    for i in range(0, len(datos)):
#        for j in range(0, len(df_1_ranking)):
#            if df_1_ranking['Symbol'] == datos['symbol'] and datos['profit'] >0 :
#                df_1_ranking['Rank'][j] =+1 
                
            

    
#    for i in range (0,len(datos)):
#        if df_1_ranking['Symbol'] == datos['symbol'] and datos['profit'] >0 :
#            df_1_ranking['Rank'] =  count(datos['profit'],[i]) / count(datos[])
        
        
        
    
    return df_1_ranking

#%%
    
def capital_acm(datos):
    
    
    datos['capital acumulado'] = 5000 +  datos.iloc[0,17]
    i =1
    for i in range(1,len(datos) -1):
        datos.iloc[i,18] = datos.iloc[i,13] + datos.iloc[i-1,18]
        #datos.iloc[i,18] = datos.iloc[i,17] + datos.iloc[i-1,18]
#    datos['capital acumulado'] = np.ones(len(datos))*5000
#    for i in range(0,len(datos)):
#        datos.iloc[i, 18] = 5000 + datos.iloc[i, 17]
#       # datos[i]['capital acumulado'] = 5000 + datos[i]['profit_acm']
#        i =+1
 
#    datos['capital acumulado'] = 5000 + datos['profit_acm']
    
    return datos

#%%
def f_profit_diario(datos):
    dates = datos['opentime']
    dates = dates[0:9]
    timeStamps = []
    for i in datos['symbol']:
        if i not in ssymb:
            ssymb.append(i)
        
        prfit_acm_d = pd.DataFrame = {'timestamp': timeStamps, 'profit_d': prfoit_d, 'profit_acm_d': profit_acm_d}
        
        
#%%
    
def f_estadisticas_mad(datos):
    rf = 0.08
    rpmat = []
    i = 1
    for i in range(1, len(datos)):
        rp = (datos['capital acumulado'][i] - datos['capital acumulado'][i-1])/ datos['capital acumulado'][i-1]
        rpmat.append(rp)
        i =+1
    desvstd = np.std(rpmat)
    Rt = sum(rpmat)/len(rpmat)
    
    logrt = np.log(1+Rt)
    vsharpe = (logrt - rf)/ desvstd
    
    rpmatpos = []
    rpmatneg = []
    i = 0
    for i in range(0, len(rpmat)):
        if rpmat[i] > 0:
            rpmatpos.append(rpmat)
        else:
            rpmatneg.append(rpmat)
                
            
    vsortino_c = (logrt - rf)/np.std(rpmatpos)
    vsortino_v = (logrt - rf)/np.std(rpmatneg)
    vdrawdown_capi_u = 1
    vdrawdown_capi_c = 1
    
    SP = pd.read_csv(r'C:/Users/juanm/Documents/Iteso/Sem10/Trading/labWork/^GSPC.csv') 
    df_SP = pd.DataFrame(SP)       
    benchmark = df_SP['Adj Close']
    rp_benchmat = []
    i = 1
    for i in range(1,len(benchmark)):
        rpbench = (benchmark[i] - benchmark[i-1])/ benchmark[i-1]
        rp_benchmat.append(rpbench)
        
    vinformation_r = 1
    
    
    
    
    estadisticas_mad = pd.DataFrame({'metrica': ['sharpe', 'sortino_c', 'sortino_v', 'drawdown_capi_c', 'drawdown_capi_u', 'information_r'],
                     'valor': [vsharpe, vsortino_c, vsortino_v, vdrawdown_capi_c, vdrawdown_capi_u, vinformation_r], 
                     'descripcion': ['Sharpe Ratio', 'Sortino Ratio para Posiciones  de Compra', 
                                     'Sortino Ratio para Posiciones de Venta', 'DrawDown de Capital', 'DrawUp de Capital',  
                                     'Informatio Ratio']})
    
    return estadisticas_mad




































