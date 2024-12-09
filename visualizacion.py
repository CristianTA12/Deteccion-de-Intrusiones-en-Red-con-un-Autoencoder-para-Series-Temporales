import cv2
import numpy as np
import torch

def visualizar_anomalias(sequences, model, anomalies, device): #sequences: conjunto de secuencias temporales. model: modelo autoencoder entrenado para reconstruir secuencias. anomalies: índices de las secuencias que se identificaron como anomalias. device: dispositivo gpu o cpu.
    for idx in anomalies[:5]:  # Limitar a las primeras 5 anomalías detectadas, para no saturar la visualización. idx: es el indice de cada secuencia anómala en el consjunto sequences
        original = sequences[idx] #Original: Contiene la secuencia original seleccionada.
        
        
        if not isinstance(original, np.ndarray): 
            original = np.array(original, dtype=np.float32) # Asegurarse de que 'original' es un numpy array de tipo float32 y si no lo es lo convierte. para asegurar compatibilidad con pytorch y opencv.
        
        # Recontruir la secuencia a partir del modelo
        reconstructed = model(torch.tensor(original, dtype=torch.float32).to(device)).cpu().detach().numpy() #Convierte la secuencia original en un tensor pytorch flotante(torch.tensor). usamos cpu. .detach para desacoplar los gradientes.
        error = np.abs(original - reconstructed) #Calcula el error absoluto entre la secuencia original y la reconstruida.

        # Crear un gráfico de calor para el error
        heatmap = cv2.applyColorMap((error * 255 / error.max()).astype(np.uint8), cv2.COLORMAP_JET)

        # Suavizar el heatmap
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)

        # Escalar la imagen si es pequeña
        heatmap = cv2.resize(heatmap, (1200, 1000), interpolation=cv2.INTER_LINEAR)

        # Configurar ventana
        cv2.namedWindow(f"Anomalia {idx}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Anomalia {idx}", 1200, 1000)  # Tamaño ajustable inicial
        
        # Mostrar la imagen con OpenCV
        cv2.imshow(f"Anomalia {idx}", heatmap) #Muestra la imagen del gráfico de calor en una ventana tutilada "Anomalia {idx}".
        cv2.waitKey(0) #Pausa el programa hasta que s epresione una tecla.

    cv2.destroyAllWindows() #Cierra todas las ventanas abiertas por OpenCV.