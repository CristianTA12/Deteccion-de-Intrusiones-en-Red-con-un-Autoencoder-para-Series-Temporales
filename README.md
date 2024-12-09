El objetivo de este proyecto es la detección de intrusiones en red a partir del entrenamiento de un modelo a través de un autoencoder. Este proyecto se divide en las siguientes fases:

1. Cargar y Procesar el Dataset:
El dataset es cargado, se procesan y limpian las columnas, y se codifican las variables categóricas.

2. Crear Ventanas Temporales:
El dataset se divide en secuencias temporales de tamaño definido por window_size.

3. Entrenar el Autoencoder:
El autoencoder es entrenado para aprender a reconstruir las secuencias de datos.

4. Detectar Anomalías:
Se calculan los errores de reconstrucción para identificar secuencias anómalas basadas en un umbral.

5. Visualizar Anomalías:
Las anomalías detectadas se visualizan utilizando un gráfico de calor.

El programa completo se ejecuta desde main.py incluyendo el entrenamiento y exportación del modelo.
En caso de tener el .pt del modelo, no necesitaremos entrenarlo, por lo tanto no será necesario ejecutar main.py y en su lugar deberemos ejecutar cargar_modelo.py, dicho archivo se encargará de cargar y procesar el dataset, cargará el modelo, detectará las anomalías y nos las mostrará en un gráfico de calor.

Notas de v1:
-Comentarios descriptivos de líneas de código.

-Ruta absoluta cambiada a ruta relativa para la correcta ejecución sin necesidad de cambiar código.

-Se incluye el archivo .csv del dataset utilizado.

-El entrenamiento ahora guarda el modelo para su utilización mas adelante.

-Se incluye el modelo entrenado.

