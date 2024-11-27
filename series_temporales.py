import numpy as np


def crear_ventanas_temporales(df, window_size): #df: La entrada puede ser un dataframe o un arreglo numpy. window_size: es el tamaño de las ventanas deslizantes.
    print('Creando ventanas temporales...')
    sequences = [] #Se crea una lista vacia llamada sequences, donde se almacenarán las ventanas temporales
    for i in range(len(df) - window_size): #Se recorre el dataframe hasta que queden suficientes filas para formar una ventana completa.
        sequence = df[i:i + window_size] #En cada iteracion extrae una ventana de tamaño window_size, comenzando en el indice i.
        sequences.append(sequence) #Agrega esta ventana a la lista de sequences.
    print(f"se han creado {len(sequences)} secuencias temporales.")
    return np.array(sequences, dtype=np.float32) #Convierte la lista sequences en un arreglo numpy con tipo de datos float32 para garantizar eficiencia y compatibilidad con modelos ML.