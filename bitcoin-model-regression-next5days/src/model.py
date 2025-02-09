import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Classic sine/cosine positional encoding for sequence data,
    as described in "Attention is All You Need" (Vaswani et al.).
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer so it's not a learnable parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        We'll add positional encoding to each batch & timestep in x.
        """
        seq_len = x.size(1)
        # Add the encoding as a constant
        x = x + self.pe[:seq_len, :]
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_size = 21,        # number of features per timestep
        d_model=64,        # dimension of the model
        nhead=8,           # number of attention heads
        num_layers=2,      # number of Transformer encoder layers
        dim_feedforward=128,
        dropout=0.3
    ):
        super().__init__()
        
        self.d_model = d_model

        # 1) Project input features -> d_model
        self.input_embedding = nn.Linear(input_size, d_model)

        # 2) Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # 3) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important if your data is (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4) Final regression head
        self.fc_out = nn.Linear(d_model, 7)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_size)

        Returns: (batch_size, 1) -> predicted value for each sample
        """
        # 1) Embed input
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)

        # 2) Add positional encoding
        x = self.pos_encoding(x)     # (batch_size, seq_len, d_model)

        # 3) Pass through Transformer Encoder
        # output shape: (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)

        # 4) We take the last timestep's output for regression
        # x[:, -1, :] => shape (batch_size, d_model)
        out = self.fc_out(x[:, -1, :])  # (batch_size, 1)
        return out