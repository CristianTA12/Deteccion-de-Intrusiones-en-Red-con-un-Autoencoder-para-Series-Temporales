import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( #Reduce gradualmente la dimensionalidad del dato de entrada.
            nn.Linear(input_size, 64), #Capa 1 de imput size a 64.
            nn.ReLU(),
            nn.Linear(64, 32), #Capa 2 de 64 a 32.
            nn.ReLU(), #Función de activación que introduce la no linealidad, ayudando al modelo a capturar relaciones complejas en los datos.
            nn.Linear(32, 16) #Capa 3 de 32 a 16.
        )
        self.decoder = nn.Sequential( #Reconstruye los datos a partir de la representación comprimida de 16 dimensiones.
            nn.Linear(16, 32), #Capa 1 de 16 a 32.
            nn.ReLU(),
            nn.Linear(32, 64), #Capa 2 de 32 a 64.
            nn.ReLU(),
            nn.Linear(64, input_size) #Capa 3 de 64 a imput size.
        )

    def forward(self, x):
        x = self.encoder(x) #Codifica el dato original en una repreentación copmprimida.
        x = self.decoder(x) #Reconstruye los datos originales desde la representación comprimida.
        return x #Retorna el dato reconstruido.



def entrenar_autoencoder(model, sequences, device, epochs=10, lr=0.0001): #Model:instancia del autoencoder. sequences: secuencias temporales a entrenar. device: GPU o CPU. epochs: numero de iteraciones sobre el cojunto de datos. lr: tasa de aprendizaje del optimizador.
    print("Comenzando entrenamiento del autoencoder...")
    
    model.to(device)  # Mueve el modelo al dispositivo correcto
    criterion = nn.MSELoss() #Mide el error cuadrátic medio entre los datos originales y reconstruidos.
    optimizer = optim.Adam(model.parameters(), lr=lr) #Optimizador que ajusta los pesos del modelo.

    sequences_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)  # Asegurarse de que las secuencias sean float32 y los mueve al dispositivo.
    for epoch in range(epochs):
        total_loss = 0.0
        for sequence in sequences_tensor:
            optimizer.zero_grad() #Reinicia los gradientes.
            output = model(sequence) #Pasa la secuencia por el modelo.
            loss = criterion(output, sequence) #Calcula la pérdida.
            loss.backward() #Retropropaga el error.
            optimizer.step() #Actualiza los pesos.
            total_loss += loss.item() #Acumula la pérdida total.
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(sequences_tensor):.6f}") #Imprime la pérdida promedio al final de cada epoch.
    path = 'modelo.pt'
    try:
        torch.save(model.state_dict(), path)
        print("Modelo guardado exitosamente en:", path)

        # Intentar cargar el modelo para verificar
        modelo_cargado = Autoencoder(input_size=sequences_tensor.shape[1])
        modelo_cargado.load_state_dict(torch.load(path))
        modelo_cargado.eval()
        print("El modelo se guardó y cargó correctamente.")
    except Exception as e:
        print(f"Error al guardar o cargar el modelo: {e}")
    return model #Retorna el modelo entrenado.


def calcular_errores(model, sequences, device): #Model: modelo autoencoder entrenado, sequences: secuencias temporales para evaluar. device: gpu o cpu.
    criterion = nn.MSELoss() #Inicializa el criterio de pérdida
    sequences_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)  # Asegurarse de que las secuencias sean float32 y los mueve al dispositivo.
    reconstruction_errors = []

    with torch.no_grad(): #Evaluar sin gradientes.
        for sequence in sequences_tensor:
            output = model(sequence) #Reconstruye con el modelo.
            error = criterion(output, sequence).item() #Calcula el error de reconstrucción.
            reconstruction_errors.append(error) #Guarda el error en la lista.

    return reconstruction_errors #Retorna la lista de errores para todas las secuencias.