import torch #Biblioteca para construir y entrenar modelos de aprendizaje profundo.
from dataset import cargar_y_procesar_dataset
from series_temporales import crear_ventanas_temporales
from autoencoder import Autoencoder, entrenar_autoencoder, calcular_errores
from visualizacion import visualizar_anomalias
import numpy as np

# Ruta del dataset
ruta = 'C:\\Proyecto\\dataset\\csv\\GeneratedLabelledFlows\\TrafficLabelling\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'

# 1. Cargar y procesar el dataset
df_scaled = cargar_y_procesar_dataset(ruta)

# 2. Crear ventanas temporales
window_size = 10
sequences = crear_ventanas_temporales(df_scaled, window_size)

# 3. Configurar y entrenar el autoencoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Detecta si hay GPU disponible y la selecciona en caso de haberlo.
print(f"Usando {'GPU' if torch.cuda.is_available() else 'CPU'}") #Imprime el dispositivo que se va a utilizar.
input_size = sequences.shape[2] #numero de características (columnas) en cada ventana.
model = Autoencoder(input_size)
model = entrenar_autoencoder(model, sequences, device)

# 4. Calcular errores y detectar anomalías
reconstruction_errors = calcular_errores(model, sequences, device)
threshold = np.percentile(reconstruction_errors, 95)
anomalies = [i for i, error in enumerate(reconstruction_errors) if error > threshold]

print(f"Umbral de anomalías: {threshold:.6f}")
print(f"Número de anomalías detectadas: {len(anomalies)}")

# 5. Visualizar anomalías detectadas
visualizar_anomalias(sequences, model, anomalies, device)