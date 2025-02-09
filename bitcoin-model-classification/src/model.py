import torch.nn as nn 

class model_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout = 0.1)
        self.fc = nn.Linear(hidden_size, 1) #fully connected layer

    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]
        out = self.fc(x)

        return out

