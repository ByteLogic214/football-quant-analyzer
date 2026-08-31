import torch
import torch.nn as nn

class LSTMMomentumNet(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 3):
        """
        Entrada: Secuencia temporal de partidos [xG, xGA, SoT, SoTA, Córners, Córners_Contra]
        Salida: Predicción de [xG, Tiros a Puerta (SoT), Córners]
        """
        super(LSTMMomentumNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # Capa de aceleración táctica (Momentum de rendimiento)
        self.momentum_dense = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU()
        )
        
        # Red de regresión no lineal
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.Softplus() # Fuerza valores estrictamente positivos
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq shape: [Batch, Seq_Len, Input_Dim]
        _, (hn, _) = self.lstm(x_seq)
        last_hidden = hn[-1]
        
        # Cálculo de Momentum = Desviación del último partido respecto a la media de la secuencia
        seq_mean = torch.mean(x_seq, dim=1)
        momentum_vector = x_seq[:, -1, :] - seq_mean
        momentum_out = self.momentum_dense(momentum_vector)
        
        # Combinar memoria recurrente + vector de aceleración
        combined = torch.cat([last_hidden, momentum_out], dim=1)
        predictions = self.fc_head(combined)
        return predictions
