import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



def entrenar_autoencoder(model, sequences, device, epochs=2, lr=0.0001):
    print("Comenzando entrenamiento del autoencoder...")
    
    model.to(device)  # Mueve el modelo al dispositivo correcto
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    sequences_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)  # Asegurarse de que las secuencias sean float32
    for epoch in range(epochs):
        total_loss = 0.0
        for sequence in sequences_tensor:
            optimizer.zero_grad()
            output = model(sequence)
            loss = criterion(output, sequence)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(sequences_tensor):.6f}")
    
    return model


def calcular_errores(model, sequences, device):
    criterion = nn.MSELoss()
    sequences_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)  # Asegurarse de que las secuencias sean float32
    reconstruction_errors = []

    with torch.no_grad():
        for sequence in sequences_tensor:
            output = model(sequence)
            error = criterion(output, sequence).item()
            reconstruction_errors.append(error)

    return reconstruction_errors