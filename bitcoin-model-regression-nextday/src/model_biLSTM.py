import torch
import torch.nn as nn
from config import INPUT_SIZE, HIDDEN_SIZE, DROPOUT_RATE, LEARNING_RATE


#model
class BitcoinBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super().__init__()
        # First Bi-LSTM + Dropout
        self.bilstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second Bi-LSTM + Dropout
        self.bilstm2 = nn.LSTM(
            input_size=hidden_size*2,  # Bidirectional doubles output size
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Final regression layer
        self.fc = nn.Linear(hidden_size*2, 1)  # Single output neuron
        
        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Layer 1
        x, _ = self.bilstm1(x)
        x = self.dropout1(x)
        x = self.relu(x)  # ReLU after first layer
        
        # Layer 2
        x, _ = self.bilstm2(x)
        x = self.dropout2(x)
        x = self.relu(x)  # ReLU after second layer
        
        # Final prediction
        x = self.fc(x[:, -1, :])  # Use last timestep's output
        return x


#criterion

class HyperbolicCosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, preds, targets):
        return torch.mean(torch.cosh(preds - targets))

#optimizer

def AdamW_optimizer(model, lr = 0.001):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)