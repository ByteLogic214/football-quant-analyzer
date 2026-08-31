import torch
import torch.nn as nn

class LSTMMomentumNet(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 3):
        super(LSTMMomentumNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # Módulo de aceleración temporal (Momentum)
        self.momentum_dense = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU()
        )
        
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.Softplus()
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x_seq)
        last_hidden = hn[-1]
        
        # Diferencial de momentum respecto a la media de la secuencia de partidos
        seq_mean = torch.mean(x_seq, dim=1)
        momentum = x_seq[:, -1, :] - seq_mean
        momentum_feats = self.momentum_dense(momentum)
        
        combined = torch.cat([last_hidden, momentum_feats], dim=1)
        return self.fc_head(combined)
