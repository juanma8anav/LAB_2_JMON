import funciones as fn                              # Para procesamiento de datos
import visualizaciones as vs                        # Para visualizacion de datos
import pandas as pd                                 # Procesamiento de datos
from datos import OA_Ak                             # Importar token para API de OANDA
import numpy as np
from matplotlib.pyplot import hist

OA_In = "EUR_USD"                  # Instrumento
OA_Gn = "D"                        # Granularidad de velas
fini = pd.to_datetime("2019-07-06 00:00:00").tz_localize('GMT')  # Fecha inicial
ffin = pd.to_datetime("2019-12-06 00:00:00").tz_localize('GMT')  # Fecha final

# Descargar precios masivos
df_pe = fn.f_precios_masivos(p0_fini=fini, p1_ffin=ffin, p2_gran=OA_Gn,
                             p3_inst=OA_In, p4_oatk=OA_Ak, p5_ginc=4900)


vs_grafica1 = vs.g_velas(p0_de=df_pe.iloc[0:120, :])
vs_grafica1.show()

# multiplicador de precios

pip_mult = 10000



# -- 0A.1: Hora

df_pe['hora'] = [df_pe['TimeStamp'][i].hour for i in range(0, len(df_pe['TimeStamp']))]



# -- 0A.2: Dia de la semana.

df_pe['dia'] = [df_pe['TimeStamp'][i].weekday() for i in range(0, len(df_pe['TimeStamp']))]

#---0A.3: Mes
df_pe['mes'] = [df_pe['TimeStamp'][i].month for i in range(0, len(df_pe['TimeStamp']))]

america = [17, 18, 19, 20, 21]
asia = [22, 23, 0, 1, 2, 3, 4, 5, 6, 7]
europa = [ 9, 10, 11, 12]
eurpa_america =[13, 14, 15, 16]

def sesion(hora):
    if hora in asia:
        x = 'Asia'
    elif hora in america:
        x = 'America'
    elif hora in europa:
        x = 'Europa'
    elif hora in eurpa_america:
        x = 'Europa_America'
    else:
        x = 'europa_asia'
    return x

df_pe['Sesion'] = [sesion(int(df_pe['hora'][i])) for i in range(0, len(df_pe['hora']))]
df_pe['Sesion']

Close = pd.DataFrame(float(i) for i in df_pe['Close'])

Open = pd.DataFrame(float(i) for i in df_pe['Open'])

DifOyC = (Close - Open)*10000
DifOyC = pd.DataFrame(DifOyC)
DifOyC

High = pd.DataFrame(float(i) for i in df_pe['High'])

Low = pd.DataFrame(float(i) for i in df_pe['Low'])

DifHyL = (High - Low)*10000
DifHyL  =pd.DataFrame(DifHyL)
DifHyL

sentido = (lambda Open, Close: 'ALCISTA' if Close >= Open else 'BAJISTA')

df_pe['Sentido'] = pd.DataFrame(sentido(df_pe['Open'][i], df_pe['Close'][i]) for i in range(len(df_pe['Open'])))


df_pe ['Sentido']


df_pe.head()

df_pe ['Sentido'].value_counts()

hist(df_pe['Sentido'])
plt.title("Sentido de las velas")
plt.ylabel("Número de repeticiones")
plt.xlabel("Sentido")
plt.show()


