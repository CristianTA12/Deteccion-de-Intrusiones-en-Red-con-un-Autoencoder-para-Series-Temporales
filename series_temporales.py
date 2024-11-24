import numpy as np


def crear_ventanas_temporales(df, window_size):
    print('Creando ventanas temporales...')
    sequences = []
    for i in range(len(df) - window_size):
        sequence = df[i:i + window_size]
        sequences.append(sequence)
    print(f"se han creado {len(sequences)} secuencias temporales.")
    return np.array(sequences, dtype=np.float32)