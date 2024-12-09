import torch #Biblioteca para construir y entrenar modelos de aprendizaje profundo.
from dataset import cargar_y_procesar_dataset
from series_temporales import crear_ventanas_temporales
from autoencoder import Autoencoder, entrenar_autoencoder, calcular_errores
from visualizacion import visualizar_anomalias
import numpy as np
from sklearn.model_selection import train_test_split

# Ruta del dataset
ruta = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'

# 1. Cargar y procesar el dataset
df_scaled = cargar_y_procesar_dataset(ruta)

# 2. Crear ventanas temporales
window_size = 10
sequences = crear_ventanas_temporales(df_scaled, window_size)

# 3. Dividir datos en entrenamiento, validación y prueba
train_val_sequences, test_sequences = train_test_split(sequences, test_size=0.2, random_state=42)
train_sequences, val_sequences = train_test_split(train_val_sequences, test_size=0.25, random_state=42)

# 4. Configurar y entrenar el modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Detecta si hay GPU disponible y la selecciona en caso de haberlo.
print(f"Usando {'GPU' if torch.cuda.is_available() else 'CPU'}") #Imprime el dispositivo que se va a utilizar.
input_size = sequences.shape[2] #numero de características (columnas) en cada ventana.
model = Autoencoder(input_size)
model = entrenar_autoencoder(model, sequences, device) #Se entrena el modelo

# 6. Evaluar el modelo en el conjunto de validación
val_errors = calcular_errores(model, val_sequences, device)
val_loss = np.mean(val_errors)
print(f"Pérdida de validación: {val_loss:.6f}")

# 7. Evaluar el modelo en el conjunto de prueba
test_errors = calcular_errores(model, test_sequences, device)
test_loss = np.mean(test_errors)
print(f"Pérdida en prueba: {test_loss:.6f}")

# 5. Calcular errores y detectar anomalías
threshold = np.percentile(test_errors, 95)
anomalies = [i for i, error in enumerate(test_errors) if error > threshold]

print(f"Umbral de anomalías: {threshold:.6f}")
print(f"Número de anomalías detectadas: {len(anomalies)}")

# 6. Visualizar anomalías detectadas
visualizar_anomalias(sequences, model, anomalies, device)