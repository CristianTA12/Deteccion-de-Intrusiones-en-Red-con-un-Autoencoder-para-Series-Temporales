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
Esto recorrerá todos los pasos anteriormente mencionados: 
Carga el dataset, que es el achivo .csv que podemos encontrar adjunto en este repositorio, con el nombre de Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv y le aplicará un eliminado de espacios en blanco, codificación de variables categóricas, reemplazará valores infinitos y nulos y escalará los datos.
Después crearemos ventanas temporales para dividir los datos en segmentos y así facilitar el aprendizaje del modelo.
El siguiente paso que va a seguir el programa es el de definir el autoencoder y entrenarlo, esta es la parte del proceso mas larga y es por ello que una vez tenemos el modelo entrenado, lo guardaremos y mas adelante si debemos ejecutar el programa nuevaamente lo haremos desde cargar_modelo.py. La última parte que se va a ejecutar en nuestro código es la de visualización mediante openCV de las anomalías que nuestro modelo ha sido capaz de encontrar.
En caso de tener el .pt del modelo, no necesitaremos entrenarlo, por lo tanto no será necesario ejecutar main.py y en su lugar deberemos ejecutar cargar_modelo.py, dicho archivo se encargará de cargar y procesar el dataset, cargará el modelo, detectará las anomalías y nos las mostrará en un gráfico de calor.


Notas de v1.5:
-Comentarios descriptivos de líneas de código.

-Ruta absoluta cambiada a ruta relativa para la correcta ejecución sin necesidad de cambiar código.

-Se incluye el archivo .csv del dataset utilizado.

-El entrenamiento ahora guarda el modelo para su utilización mas adelante.

-Se incluye el modelo entrenado.

-Se añade funcionalidad a main.py y cargar_modelo.py para dividir los datos en: datos de entrenamiento, datos de validación y datos de prueba.

-Asimismo al ejecutar el programa obtenemos información nueva como: Pérdida de validación y pérdida de prueba.

-Los porcentajes para entrenamiento, validación y prueba son los siguientes: 60% entrenamiento, 20% valdiación, 20% pruebas.
