import cv2
import numpy as np
import torch

def visualizar_anomalias(sequences, model, anomalies, device):
    for idx in anomalies[:5]:  # Limitar a las primeras 5 anomalías detectadas
        original = sequences[idx]
        
        # Asegurarse de que 'original' es un numpy array de tipo float32
        if not isinstance(original, np.ndarray):
            original = np.array(original, dtype=np.float32)

        # Trasladar 'original' a la GPU (si está disponible)
        original_tensor = torch.tensor(original, dtype=torch.float32).to(device)

        # Hacer la inferencia en la GPU y luego mover la salida de vuelta a la CPU
        reconstructed = model(original_tensor).cpu().detach().numpy()
        
        # Calcular el error
        error = np.abs(original - reconstructed)

        # Crear un gráfico de calor para el error
        heatmap = cv2.applyColorMap((error * 255 / error.max()).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Mostrar la imagen con OpenCV
        cv2.imshow(f"Anomalía {idx}", heatmap)
        cv2.waitKey(0)

    cv2.destroyAllWindows()