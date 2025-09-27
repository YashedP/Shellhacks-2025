import torch
import torch.nn as nn

class CoastlineLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=2):
        super(CoastlineLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)            # [batch, seq_len, hidden]
        out = self.fc(out[:, -1, :])     # predict from last timestep
        return out

# Example
model = CoastlineLSTM()
x = torch.randn(32, 20, 3)  # batch=32, seq_len=20 years, features=(lat, lon, year)
y = model(x)
print(y.shape)  # [32, 2] -> predicted (lat, lon)
