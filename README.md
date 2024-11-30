Uso
1. Cargar y Procesar el Dataset
El dataset es cargado, se procesan las columnas, y se codifican las variables categóricas.

2. Crear Ventanas Temporales
El dataset se divide en secuencias temporales de tamaño definido por window_size.

3. Entrenar el Autoencoder
El autoencoder es entrenado para aprender a reconstruir las secuencias de datos.

4. Detectar Anomalías
Se calculan los errores de reconstrucción para identificar secuencias anómalas basadas en un umbral.

5. Visualizar Anomalías
Las anomalías detectadas se visualizan utilizando un gráfico de calor.

El programa completo se ejecuta desde main.py

Notas de v1:
-Comentarios descriptivos de líneas de código.
-Ruta absoluta canbiada a ruta relativa para la correcta ejecución sin necesidad de cambiar código.
-Se incluye el archivo .csv del dataset utilizado.
