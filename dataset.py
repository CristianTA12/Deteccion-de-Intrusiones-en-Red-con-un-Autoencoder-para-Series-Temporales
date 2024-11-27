import pandas as pd #Para trabajar con datasets (lectura, manipulación, análisis)
import numpy as np #Para operaciones matemáticas y manejo de arrays.
from sklearn.preprocessing import MinMaxScaler #Para escalar los datos numéricos a un rango entre o y 1.

def cargar_y_procesar_dataset(ruta):
    print('Cargando y procesando dataset...')
    df = pd.read_csv(ruta)
    print('Dataset cargado correctamente.')

    print('Eliminando espacios en los nombres de las columnas...')
    df.columns = df.columns.str.strip() #Elimina espacios en blanco al principio y al final de los nombres de las columnas.
    print(df.info()) #Muestra información sobre las columnas, sus tipos de datos y valores no nulos.
    print(df.describe()) #Muestra estadísticas descrptivas (media, desviación estándar, etc... De las columnas numéricas.)

    #Codificaión de variables categóricas
    df = pd.get_dummies(df, columns=['Protocol', 'Label']) #Transforma las columnas categóricas Protocol y Label en variables dummy; codificaión one-hot.

    #conversión de Timestamp y ordenamiento
    df['Timestamp'] = pd.to_datetime(df['Timestamp']) #Convierte la columna Timestamp al formato datetime.
    df = df.sort_values(by='Timestamp') # Ordena el dataset cronológicamente usando timestamp.
    
    #Reemplazar valores infinitos y NaN
    df = df.select_dtypes(include=[np.number]) #Selecciona únicamente las columnas numéricas del dataframe, esto descarta cualquier columna categórica o de texto.
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0) #Reemplaza valores infinitos con valores nulos y llena los valores nulos con 0, asegurando que no queden datos faltantes.

    #Escalar los datos con MinMaxScaler
    print('Escalando los datos...')
    scaler = MinMaxScaler() #Escala todas las columnas numericas al rango [0,1] utilizando MinMaxScaler
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns) #Crea un nuevo dataframe con los datos escalados y las mismas columnas originales.
    print('Datos escalados...')
    
    return df_scaled #Devuelve dataset procesado y escalado.