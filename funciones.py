
# -- ------------------------------------------------------------------------------------ -- #
# -- Proyecto: Describir brevemente el proyecto en general                                -- #
# -- Codigo: RepasoPython.py - describir brevemente el codigo                             -- #
# -- Repositorio: https://github.com/                                                     -- #
# -- Autor: Juan Mario Ochoa Navarro                                                      -- #
# -- ------------------------------------------------------------------------------------ -- #

import numpy as np                                      # funciones numericas
import pandas as pd                                       # dataframes y utilidades
from datetime import timedelta                            # diferencia entre datos tipo tiempo
import datetime as datetime
from  datetime import date, timedelta
import yfinance as yf 
import plotly.offline as py  
py.offline.init_notebook_mode

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
    #df_data = pd.read_csv(r'C:\Users\Usuario\Documents\Sem9\Trading\labWork\' + param_archivo)#, sheet_name='archivo_tradeview_1') 
    df_data = pd.read_csv(r'C:/Users/Usuario/Documents/Sem9/Trading/labWork/' + param_archivo)#, sheet_name='archivo_tradeview_1')
   # Convertir a minusculas el nombre de las columnas
    df_data.columns = [list(df_data.columns)[i].lower() for i in range(0, len(df_data.columns))]
    # Asegurar que ciertas columnas son tipo numerico
    numcols =  ['s/l', 't/p', 'comission', 'openprice', 'closeprice', 'profit', 'size', 'swap', 'taxes']
    df_data[numcols] = df_data[numcols].apply(pd.to_numeric)  
    #datcols = ['opentime', 'closetime']
    #df_data[datcols] = df_data[datcols].apply(pd.to_datetime)
    #del df_data['balance']
    df_data['symbol'].str.lower()
    df_data['symbol'].replace('/', '') 
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
    #param_ins =  'btcusd'
    
    # encontrar y eliminar un guion bajo
    inst = param_ins.replace('_', '')
    inst = param_ins.replace('-2', '')
    inst = param_ins.replace('/', '')    
    # transformar a minusculas
    inst = inst.lower()
    
    #lista de pips por instrumento
    pip_inst = {'usdjpy': 10000, 'gbpjpy': 10000, 'eurjpy': 10000, 'cadjpy': 10000, 'usdczk':10000,
                'chfjpy': 10000,'eurusd': 10000, 'gbpusd': 10000, 'usdcad': 10000, 'usddkk':10000,
                'usdmxn': 10000,'audusd': 10000, 'nzdusd': 10000, 'usdchf': 10000, 'usdcnh':10000,
                'eurgbp': 10000, 'eurchf': 10000, 'eurnzd': 10000, 'euraud': 10000, 'usdzar': 10000, 
                'gbpnzd': 10000, 'gbpchf': 10000, 'gbpaud': 10000, 'audnzd': 10000, 
                'nzdcad': 10000,'audcad': 10000, 'xauusd': 10, 'xagusd': 10, 'btcusd': 1,
                'nas100usd': 10, 'us30usd': 10, 'mbtcusd':100, 'usdmxn': 10000}
       
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
    
    datos['closetime'] = pd.to_datetime( datos['closetime'])
    datos['opentime'] = pd.to_datetime( datos['opentime'])
    
    datos['tiempo'] = [(datos.loc[i, 'closetime'] - datos.loc[i, 'opentime']).delta / 1*np.exp(9) for i in range(0, len(datos))]
    #todos van a dar 0 ya que las operaciones se cerraron el mismo día que se abrieron     
    return datos

#%%
def f_columnas_pips(datos):
    datos['pips'] = np.zeros(len(datos))
    i=0
    for i in range(0, len(datos)):
        #if datos.iloc[2, i] == "buy":
        if datos['type'][i] == 'buy':
            datos['pips'][i] = (datos.closeprice[i] - datos.openprice[i])*f_pip_size(datos['symbol'][i])
        else:
            datos['pips'][i] = (datos.closeprice[i] - datos.openprice[i])*f_pip_size(datos['symbol'][i])*-1
            #datos['pips'][i] = (datos.openprice[i] - datos.closeprice[i])*f_pip_size(datos['symbol'][i])
            
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
   
    v3 = len(datos[(datos.type == 'sell') & (datos.profit > 0)])
        
    v4 = v0-v1
    
    v5 = len(datos[(datos.type == 'buy') & (datos.profit < 0)])
    
    v6 = len(datos[(datos.type == 'sell') & (datos.profit < 0)])
        
    v7 = np.median(datos.profit)
    
    v8 = np.median(datos.pips)
    
    v9 =v1/v0
    
    v10 = v1/v4
    
    v11 = v2/v0
    
    v12 = v3/v0
    
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
    
    ssymb = datos["symbol"].unique().tolist()

    rnk = np.zeros(shape = (len(ssymb),2))

    df_1_ranking = pd.DataFrame(rnk, columns = ['Symbol' , 'Rank'])
     
    for i in range (0,len(ssymb)):
        df_1_ranking['Symbol'][i] = ssymb[i]
        
    for i in range (0,len(df_1_ranking["Symbol"])):
        g = 0
        t = 0
        for j in range (0,len(datos["symbol"])):
            if df_1_ranking["Symbol"][i] == datos["symbol"][j]:
                t =t + 1 
                if datos['profit'][j] > 0:
                    g =g+1
                
        df_1_ranking['Rank'][i] = g/t

    



    
    
    
    
    
    #ssymbol = datos['symbol'].name.unique
    
#    ssymb = []
#    for i in datos['symbol']:
#        if i not in ssymb:
#            ssymb.append(i)
#            
#    rnk = np.zeros(len(ssymb))
#    
#    
#    
#    df_1_ranking = pd.DataFrame({'Symbol' : ssymb, 'Rank' : rnk})
    

    
#    g = 0
#    t = 0
#    for i in range(0,len(df_1_ranking)):
#        for j in range(0, len(datos)):
#            if df_1_ranking['Symbol'][i] == datos['symbol'][j]:
#                t =+1
#                if datos['profit'][j] > 0:
#                    g =+1
#        df_1_ranking['Rank'][i] = g/t
#        
        
    
    
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
    
    
    #datos['capital_acm'] = 5000 +  datos.iloc[0,17]
#    i =1
#    for i in range(1,len(datos)):
        
        #datos.iloc[i,18] = datos.iloc[i,13] + datos.iloc[i-1,18]
        #datos.iloc[i,18] = datos.iloc[i,17] + datos.iloc[i-1,18]
#    datos['capital acumulado'] = np.ones(len(datos))*5000
#    for i in range(0,len(datos)):
#        datos.iloc[i, 18] = 5000 + datos.iloc[i, 17]
#       # datos[i]['capital acumulado'] = 5000 + datos[i]['profit_acm']
#        i =+1
 
    datos['capital_acm'] = 5000 + datos['profit_acm']
    
    return datos

#%%
def f_profit_diario(datos):
    #profit_d = profit_acm
    #profit_acm_d = capital_acm
    from datetime import date
    import yfinance as yf 
    capital = 5000
    
 
    
        
    
    
    
    
    
    
    
    #dummies ts
    #tsdummies = pd.get_dummies(datos["opentime"])
    #tsnum = np.zeros(len(datos["opentime"]))
    
#    for i in range(0,len(tsnum)):
#        sum(1 in tsnum[i], if) 
    
    
    
    #from datetime import datetime, timedelta
    ts = datos["opentime"].unique().tolist()#.date() #timestamp
    
#    ts = []
#    for i in datos['openprice']:
#        if i not in ts:
#            ts.append(i)
    
    
#    ssymb = []
#    for i in datos['symbol']:
#        if i not in ssymb:
#            ssymb.append(i)
    
    #ts = datetime(year = int(prets[0:4]), month = int(prets[4:6]), day = int(prets[6:8]))

    #pro = np.zeros(shape = (len(ts),3)) #matriz vacía
    
    
    
    #modificamos lasfechaspor números para comparar, este es del histórico
#    j = 1
#    matd= np.zeros(len(datos["opentime"]))
#    matd[0] = 1
#    for i in range(1,len(datos["opentime"])):
#        matd[i] = 1
#        if datos['opentime'][i] == datos['opentime'][i-1]:
#            i = i+1
#    #lista de uniques para los dias y sus profits        
#    ts = []
#    for i in matd:
#        if i not in ts:
#            ts.append(i)       
    
    pro = np.zeros(shape = (len(ts),3))

    df_profit_diario = pd.DataFrame(pro, columns = ['timestamp' , 'profit_d', 'profit_acm_d'])
    
    
    #comparador = {'tiempo':datos["opentime"], 'prfofit_acum': datos['profit_acm']} 
#    comparador = []
#    comparador['tiempo'] = datos.iloc['opentime']
#    comparador['profit_acum'] = datos['capital_acm']
    
    comp = np.zeros(shape = (len(datos['opentime']),2))
    
#    df_comparador = pd.DataFrame(comp, columns = ['tiempo', 'profit_acum'])
#    for i in range(0,len(datos['opentime'])):
#        df_comparador['tiempo'][i] = matd[i]
#        df_comparador['profit_acum'][i] = datos['profit_acm'][i]
        
    #df_comparador['tiempo']
    
    for i in range (0,len(ts)):
        df_profit_diario['timestamp'][i] = ts[i] #asignamos timestamp
        
    #tscomp = datos['opentime']#.apply(pd.to_numeric)
        #df_data[numcols] = df_data[numcols].apply(pd.to_numeric)
        
    #df_profit_diario['profit_d'].apply(pd.to_numeric)
    #datos['profit_acm'].apply(pd.to_numeric)
    #N=np.floor(np.divide(1,delta))
 #============================================================================================================       
        #i será el contador grande
        #j sera el contador chico
#    for i in range(0,len(matd)):
#        for j in range(0,len(df_profit_diario['timestamp'])):
#            if matd[i] == ts[j]:
#                df_profit_diario['profit_d'][j].sum(datos['profit_acm'][i])
                
        
#    for i in range (0,len(df_profit_diario["timestamp"])):
#        
#        for j in range (0,len(datos["opentime"])):
#            if df_profit_diario["timestamp"][i] == datos["opentime"][j]:
#                df_profit_diario['profit_d'][i].sum(datos['profit_acm'][j])    
                
            
                
        #['Rank'][i] = g/t
    
#    s_date = datos['closetime'][0].date()
#    e_date = datos['closetime'][len(datos)-1].date()
#    
#    Δ = e_date - s_date 
#    
#    sp500 = yf.download('^gspc', 
#                     start=s_date, 
#                     end=e_date, 
#                     progress=False)
#    
#    sp500.head()
#    sp500 = sp500.reset_index()
#    
#    sp500['Rendimientos Log'] = np.zeros(len(sp500))#['Adj Close']))
#    for i in range (1,len(sp500)):#['Adj Close'])):
#        sp500['Rendimientos Log'][i] = np.log(sp500['Adj Close'][i]/sp500['Adj Close'][i-1])
#        
#    
#    relleno1 = np.zeros(shape=(Δ.days+1,6))
#    df_profit_gen = pd.DataFrame(relleno1, columns = ['Timestamp','Profit Diario',
#                                                     'Capital Acumulado', 'Rendimientos Log','Rend Log SP',
#                                                     'Traceback Error'])
#    relleno2 = np.zeros(shape=(Δ.days+1,4))    
#    df_profit_compra = pd.DataFrame(relleno2, columns = ['Timestamp','Profit Diario',
#                                                     'Capital Acumulado', 'Rendimientos Log'])
#    df_profit_venta = pd.DataFrame(relleno2, columns = ['Timestamp','Profit Diario',
#                                                     'Capital Acumulado', 'Rendimientos Log'])
#    
#    
#    for i in range(0,Δ.days+1):
#        df_profit_gen["Timestamp"][i]= s_date + timedelta(days=i)
#        df_profit_compra["Timestamp"][i]= s_date + timedelta(days=i)
#        df_profit_venta["Timestamp"][i]= s_date + timedelta(days=i)
#        
#        
#    for i in range (0,len(df_profit_gen["Timestamp"])):
#        a = 0
#        b = 0
#        c = 0
#        
#        for k in range (0,len(datos["closetime"])):
#            if df_profit_gen["Timestamp"][i] == datos["closetime"][k].date():
#                a = a + datos["profit"][k]
#                
#                if datos['type'][k] == 'buy':
#                    b = b + datos["profit"][k]
#                    
#                elif datos['type'][k] == 'sell':
#                    c = c + datos["profit"][k]
#            
#        df_profit_gen["Profit Diario"][i] = a
#        df_profit_compra["Profit Diario"][i] = b
#        df_profit_venta["Profit Diario"][i] = c  
#            
#    for i in range (0,len(df_profit_gen["Timestamp"])):
#        
#        for k in range (0,len(sp500["Date"])):
#            if df_profit_gen["Timestamp"][i] == sp500["Date"][k].date():
#                df_profit_gen["Rend Log SP"][i] = sp500["Rendimientos Log"][k]
#                
#        
#    df_profit_gen = df_profit_gen.sort_values(by=['Timestamp'])
#    df_profit_gen = df_profit_gen.reset_index(drop=True)
#    
#    df_profit_compra = df_profit_compra.sort_values(by=['Timestamp'])
#    df_profit_compra = df_profit_compra.reset_index(drop=True)
#    
#    df_profit_venta = df_profit_venta.sort_values(by=['Timestamp'])
#    df_profit_venta = df_profit_venta.reset_index(drop=True)
#    
#    df_profit_gen['Capital Acumulado'][0] = capital + df_profit_gen['Profit Diario'][0]
#    df_profit_gen['Rendimientos Log'][0] = np.log(df_profit_gen['Capital Acumulado'][0]/capital)
#    
#    df_profit_compra['Capital Acumulado'][0] = capital + df_profit_compra['Profit Diario'][0]
#    df_profit_compra['Rendimientos Log'][0] = np.log(df_profit_compra['Capital Acumulado'][0]/capital)
#    
#    df_profit_venta['Capital Acumulado'][0] = capital + df_profit_venta['Profit Diario'][0]
#    df_profit_venta['Rendimientos Log'][0] = np.log(df_profit_venta['Capital Acumulado'][0]/capital)
#            
#    for i in range(1,len(df_profit_gen["Profit Diario"])):
#         df_profit_gen['Capital Acumulado'][i] = df_profit_gen['Capital Acumulado'][i-1] + df_profit_gen['Profit Diario'][i]
#         df_profit_gen['Rendimientos Log'][i] = np.log(df_profit_gen['Capital Acumulado'][i]/df_profit_gen['Capital Acumulado'][i-1])
#         df_profit_gen["Traceback Error"][i] = df_profit_gen["Rendimientos Log"][i]-df_profit_gen["Rend Log SP"][i]
#         
#         df_profit_compra['Capital Acumulado'][i] = df_profit_compra['Capital Acumulado'][i-1] + df_profit_compra['Profit Diario'][i]
#         df_profit_compra['Rendimientos Log'][i] = np.log(df_profit_compra['Capital Acumulado'][i]/df_profit_compra['Capital Acumulado'][i-1])
#         
#         df_profit_venta['Capital Acumulado'][i] = df_profit_venta['Capital Acumulado'][i-1] + df_profit_venta['Profit Diario'][i]
#         df_profit_venta['Rendimientos Log'][i] = np.log(df_profit_venta['Capital Acumulado'][i]/df_profit_venta['Capital Acumulado'][i-1])
#         
#    m = 0
#    c = 0
#     
#    while m != 5:
#        m = df_profit_gen["Timestamp"][c].weekday()
#        c = c + 1
#    
#    lista = []
#    
#    for i in range (c-1,len(df_profit_gen["Timestamp"])+1,7):
#        
#        lista.append(i)
#        
#        
#    
#        
#    df_profit_gen = df_profit_gen.drop(df_profit_gen.index[lista])
#    df_profit_compra = df_profit_compra.drop(df_profit_compra.index[lista])
#    df_profit_venta = df_profit_venta.drop(df_profit_venta.index[lista])
    #-----------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    
    #from datetime import datetime
#    dates = datos['opentime'].unique().tolist()
#    dates.date()# = pd.to_datetime(dates)
   #dates = dates.dt.strftime(" %b %d %Y")  #'list' object has no attribute 'dt'
#    for i in range(0, len(dates)):
#        dates[i] = dates[i].dt.strftime(" %b %d %Y")
        
#    relleno1 = np.zeros(len(dates))
 #   relleno2 = np.zeros(len(dates))
    
  #  df_profit_acm_d = pd.DataFrame({'timestamp': dates, 'profit_d': relleno1,'profit_acm_d': relleno2})
    #df_profit_acm_d['timestamp'] = df_profit_acm_d['timestamp'].dt.date('%m/%d/%Y')
    
    
    #df_profit_acm_d = pd.DataFrame(relleno, columns = ['timestamp','profit_d','profit_acm_d'])
    
#    for i in range(0,len(dates)):
#        df_profit_acm_d['timestamp'][i] = dates[i]
    
    #dates = datos['opentime']
    #dates = dates[0:9]
    #timeStamps = []
#    for i in datos['symbol']:
#        if i not in ssymb:
#            ssymb.append(i)
    
        #Profit_acm_d = pd.DataFrame = {'timestamp': timeStamps, 'profit_d': profit_d, 
        #                               'profit_acm_d': profit_acm_d}
        
    #return df_profit_acm_d
    #profit_diario_acum = {"Profit acum Gral":df_profit_gen ,"Profit compra":df_profit_compra,"profit venta":df_profit_venta,"S&P500":sp500}
        
    #return profit_diario_acum
    return df_profit_diario
    
    
    
#%%
from matplotlib import pyplot    
#import statsmodel.api as sm

def f_estadisticas_mad(datos):
    rf = 0.08/300
    #mar = .3/300
    rpmat = []
    i = 1
    for i in range(1, len(datos)):
        rp = (datos['capital_acm'][i] - datos['capital_acm'][i-1])/ datos['capital_acm'][i-1]
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
            rpmatpos.append(rpmat[i])
        else:
            rpmatneg.append(rpmat[i])
                
            
    vsortino_c = (logrt - rf)/np.std(rpmatpos)
    vsortino_v = (logrt - rf)/np.std(rpmatneg)
    
    
    #Grafcamos el profit acumulado
    #datos['capital_acm'].plot()
    #decomposition = sm.tsa.seasonal_decompose(datos['capital_acm'], model = 'aditive')
    pyplot.plot(datos.index, datos['capital_acm'], c='blue')
    #pyplot.plot(decomposition.trend.index, decomposition.trend, c='red')
    pyplot.show()
    
#    if max(datos['capital_acm']).index > min(datos['capital_acm']).index:
#        p1 = base
#        p2 = max(datos['capital_acm'])
#        p3 = min(datos['capital_acm'])
#    else:
#        p1 = base
#        p2 = min(datos['capital_acm'])
#        p3 = max(datos['capital_acm'])
    
    f1dd = datos['opentime'][21]
    f2dd = datos['closetime'][24]
    difdd = datos['capital_acm'][24]-datos['capital_acm'][21]
    
    f1du = datos['opentime'][31]
    f2du = datos['closetime'][41]
    difdu = datos['capital_acm'][41]-datos['capital_acm'][31]

    
    #for i in range(0, len(datos['capital_acm'])):
    #profit_diario_acum = {"Profit acum Gral":df_profit_gen ,"Profit compra":df_profit_compra,"profit venta":df_profit_venta,"S&P500":sp500}    
    
    
    vdrawdown_capi = {'Fecha inicial': f1dd, 'Fecha Final':f2dd, 'DrawDaown$':difdd}
    vdrawup_capi = {'Fecha inicial': f1du, 'Fecha Final':f2du, 'Drawup$':difdu}
    #df_vdrawdown_capi = pd.DataFrame(vdrawdown_capi)
    #df_vdrawup_capi = pd.DataFrame(vdrawup_capi)
    
    SP = pd.read_csv(r'C:/Users/Usuario/Documents/Sem9/Trading/labWork//^GSPC.csv') 
    df_SP = pd.DataFrame(SP)       
    benchmark = df_SP['Adj Close']
    rp_benchmat = []
    i = 1
    for i in range(1,len(benchmark)):
        rpbench = (benchmark[i] - benchmark[i-1])/ benchmark[i-1]
        rp_benchmat.append(rpbench)
        
    vinformation_r = 1
    
    
    
    
    estadisticas_mad = pd.DataFrame({'metrica': ['sharpe', 'sortino_c', 'sortino_v', 'drawdown_capi_c', 'drawdown_capi_u', 'information_r'],
                     'valor': [vsharpe, vsortino_c, vsortino_v, vdrawdown_capi, vdrawup_capi, vinformation_r], 
                     'descripcion': ['Sharpe Ratio', 'Sortino Ratio para Posiciones  de Compra', 
                                     'Sortino Ratio para Posiciones de Venta', 'DrawDown de Capital', 'DrawUp de Capital',  
                                     'Informatio Ratio']})
    
    return estadisticas_mad

#%%Sesgos Cognitivos
    

#%%Gráficas
#import cufflinks as cf
#import plotly.plotly as py
##import plotly.tools as tls
#import plotly.graph_objects as go 
#
#a = np.linspace(start = 0, stop = 36, num = 36)
#
#np.random.seed(25)
#b = np.random.uniform(los = 0, high = 1, size = 36)
#
#trace = go.scatter(x=a, y=b)
#data = [trace]
#
#py.iplot(data)
#%%Gráficas
#import plotly.offline as py  
#py.offline.init_notebook_mode#(connected=False)  
#import plotly.express as px
#import plotly.graph_objects as go
#def graph1(datos):
#    labels = np.transpose(datos.iloc[:,0])
#    values = np.transpose(datos.iloc[:,1])
#     
#    # pull is given as a fraction of the pie radius
#    #fig = py.graph_objects.Figure(data=[py.graph_objects.Pie(labels=labels, values=values)])
#    fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0.2, 0])])
#    
#    py.iplot(fig)

#%%
#import plotly
#from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
#import plotly.graph_objs as go
#
#import plotly.plotly as py
#py.init_notebook_mode(connected = False)
#
#def graph1(datos):
#    labels = np.transpose(datos.iloc[:,0])
#    values = np.transpose(datos.iloc[:,1])
#     
#    # pull is given as a fraction of the pie radius
#    trace = go.Pie(labels = labels, values = values,
#                   hoverinfo = 'label + percent', textinfo = 'value',
#                   textfont = dict(size =25))
#    #fig = py.graph_objects.Figure(data=[py.graph_objects.Pie(labels=labels, values=values)])
#    #fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0.2, 0])])
#    
#    py.iplot([trace])
#%% graph2
import plotly.graph_objects as go
def graph2(input1):#, input2):
#    hist = input1['capital_acm']
    fechas = input1['closetime']
 #   drawdown = input1['capital_acm'][21:24]
  #  drawup = input1['capital_acm'][31:41]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = fechas, y = input1['capital_acm'], name = 'profit histórico',
                  line=dict(color='rgba(0, 0, 0, 0.5)', width=4)))
    
    fig.add_trace(go.Scatter(x = fechas, y = input1['capital_acm'][21:24], name = 'drawdown',
                  line=dict(color='rgba(152, 0, 0, .8)', width=4, dash = 'dot'))) #recta punteada rojo
    
    fig.add_trace(go.Scatter(x = fechas, y = input1['capital_acm'][31:41], name = 'drawup',
                  line=dict(color='rgba(30, 130, 76, 1)', width=4, dash = 'dot'))) #recta punteadaa verde
    
    fig.show()

#%%graph3
   




















