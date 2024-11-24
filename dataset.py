import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def cargar_y_procesar_dataset(ruta):
    print('Cargando y procesando dataset...')
    df = pd.read_csv(ruta)
    print('Dataset cargado correctamente.')

    print('Eliminando espacios en los nombres de las columnas...')
    df.columns = df.columns.str.strip()
    print(df.info())
    print(df.describe())

    #Codificaión de variables categóricas
    df = pd.get_dummies(df, columns=['Protocol', 'Label'])

    #conversión de Timestamp y ordenamiento
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by='Timestamp')
    
    #Reemplazar valores infinitos y NaN
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    #Escalar los datos con MinMaxScaler
    print('Escalando los datos...')
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    print('Datos escalados...')
    
    return df_scaled