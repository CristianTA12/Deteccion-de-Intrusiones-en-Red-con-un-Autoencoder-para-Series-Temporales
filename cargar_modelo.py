import torch  # Biblioteca para construir y entrenar modelos de aprendizaje profundo.
from dataset import cargar_y_procesar_dataset
from series_temporales import crear_ventanas_temporales
from autoencoder import Autoencoder, calcular_errores
from visualizacion import visualizar_anomalias
import numpy as np
import os  # Biblioteca para verificar si existe el archivo del modelo.

# Ruta del dataset
ruta = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'

# Ruta para guardar/cargar el modelo
modelo_guardado = "modelo.pt"

# 1. Cargar y procesar el dataset
df_scaled = cargar_y_procesar_dataset(ruta)

# 2. Crear ventanas temporales
window_size = 10
sequences = crear_ventanas_temporales(df_scaled, window_size)

# 3. Configurar el dispositivo y el modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando {'GPU' if torch.cuda.is_available() else 'CPU'}")

input_size = sequences.shape[2]
model = Autoencoder(input_size)

# Verificar si el modelo ya está guardado
if os.path.exists(modelo_guardado):
    # Cargar modelo guardado
    model.load_state_dict(torch.load(modelo_guardado))
    model.to(device)
    model.eval()
    print("Modelo cargado exitosamente.")
else:
    print("No se encontró un modelo guardado. Por favor, entrena el modelo primero.")
    exit()  # Salir si no se encuentra el modelo.

# 4. Calcular errores y detectar anomalías
reconstruction_errors = calcular_errores(model, sequences, device)
threshold = np.percentile(reconstruction_errors, 95)
anomalies = [i for i, error in enumerate(reconstruction_errors) if error > threshold]

print(f"Umbral de anomalías: {threshold:.6f}")
print(f"Número de anomalías detectadas: {len(anomalies)}")

# 5. Visualizar anomalías detectadas
visualizar_anomalias(sequences, model, anomalies, device)