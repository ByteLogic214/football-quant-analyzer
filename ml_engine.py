import numpy as np
import torch
from catboost import CatBoostRegressor
from lstm_momentum import LSTMMomentumNet
from typing import Dict

class HybridPredictor:
    def __init__(self):
        self.cb_xg = CatBoostRegressor(iterations=350, learning_rate=0.03, depth=6, verbose=0)
        self.cb_sot = CatBoostRegressor(iterations=350, learning_rate=0.03, depth=6, verbose=0)
        self.cb_corners = CatBoostRegressor(iterations=350, learning_rate=0.03, depth=6, verbose=0)
        
        self.lstm = LSTMMomentumNet(input_dim=6, hidden_dim=64, num_layers=2, output_dim=3)
        self.lstm.eval()
        self._fit_initial_calibration()

    def _fit_initial_calibration(self):
        """Inicializa los hiperplanos de regresión de CatBoost."""
        np.random.seed(42)
        X_calib = np.random.uniform(0.5, 90.0, (200, 8))
        y_xg = np.random.uniform(0.5, 3.2, 200)
        y_sot = np.random.uniform(2.0, 10.0, 200)
        y_cor = np.random.uniform(2.0, 9.0, 200)
        
        self.cb_xg.fit(X_calib, y_xg)
        self.cb_sot.fit(X_calib, y_sot)
        self.cb_corners.fit(X_calib, y_cor)

    def predict(self, tabular_feats: np.ndarray, sequence_feats: np.ndarray) -> Dict[str, float]:
        x_tab = tabular_feats.reshape(1, -1)
        cb_xg = float(self.cb_xg.predict(x_tab)[0])
        cb_sot = float(self.cb_sot.predict(x_tab)[0])
        cb_cor = float(self.cb_corners.predict(x_tab)[0])
        
        with torch.no_grad():
            x_seq_t = torch.tensor(sequence_feats, dtype=torch.float32).unsqueeze(0)
            lstm_out = self.lstm(x_seq_t).numpy()[0]
            lstm_xg, lstm_sot, lstm_cor = float(lstm_out[0]), float(lstm_out[1]), float(lstm_out[2])

        # Ensamble 50/50
        return {
            "xg": round(max(0.2, (0.5 * cb_xg) + (0.5 * lstm_xg)), 2),
            "sot": round(max(1.0, (0.5 * cb_sot) + (0.5 * lstm_sot)), 1),
            "corners": round(max(1.0, (0.5 * cb_cor) + (0.5 * lstm_cor)), 1)
        }
