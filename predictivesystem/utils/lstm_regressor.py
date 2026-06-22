import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_size : int, hidden_size : int, num_layers : int, dropout : float):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, (hn, _) = self.lstm(x)

        out = hn[-1]
        out = self.fc(out)

        return out.squeeze()