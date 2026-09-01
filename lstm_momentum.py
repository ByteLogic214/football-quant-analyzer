import torch
import torch.nn as nn

class LSTMMomentumNet(nn.Module):
    """
    Red LSTM con mecanismo de Momentum para predicción de métricas de fútbol.
    
    Arquitectura:
    - LSTM: Procesa secuencia temporal [batch, 5, 6]
    - Momentum: Captura aceleración (último - media)
    - Salida: [xG, SoT, Corners] con Softplus para no-negatividad
    
    Entrada (Batch, 5, 6):
        Últimos 5 partidos × [xG, xGA, SoT, SoTA, Corners, Corners_contra]
    
    Salida (Batch, 3):
        [xG_predicted, SoT_predicted, Corners_predicted] con valores ≥ 0
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 3):
        super(LSTMMomentumNet, self).__init__()
        
        # LSTM: Procesa secuencia temporal de 5 partidos
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # Módulo de Momentum: Captura aceleración (diferencial respecto a media)
        # Input: vector de 6 dimensiones (último partido - media de secuencia)
        self.momentum_dense = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Cabeza de salida: Combina LSTM hidden + momentum features
        # hidden_dim (64) + 32 (momentum) = 96
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.Softplus()  # Garantiza salidas ≥ 0
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. LSTM procesa secuencia temporal
        2. Extrae último hidden state
        3. Calcula momentum = (último partido - media de secuencia)
        4. Combina y proyecta a salida
        
        Args:
            x_seq: Tensor [batch, 5, 6]
        
        Returns:
            out: Tensor [batch, 3] con predicciones ≥ 0
        """
        # LSTM: retorna salida de todos los timesteps + hidden states finales
        _, (hn, _) = self.lstm(x_seq)
        # hn shape: [num_layers, batch, hidden_dim]
        # Extrae el último layer
        last_hidden = hn[-1]  # [batch, hidden_dim]
        
        # Calcula momentum: aceleración respecto a la media histórica
        seq_mean = torch.mean(x_seq, dim=1)  # [batch, 6]
        momentum = x_seq[:, -1, :] - seq_mean  # [batch, 6]
        momentum_feats = self.momentum_dense(momentum)  # [batch, 32]
        
        # Combina LSTM + Momentum
        combined = torch.cat([last_hidden, momentum_feats], dim=1)  # [batch, 96]
        
        # Proyecta a salida con Softplus (no-negatividad)
        return self.fc_head(combined)  # [batch, 3]
