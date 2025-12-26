"""
Transformer Model cho dự báo giá cổ phiếu
So sánh với LSTM
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional Encoding cho Transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class StockTransformer(nn.Module):
    """Transformer Model cho dự báo giá cổ phiếu"""
    
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1, forecast_days=3):
        """
        Args:
            input_size: Số lượng features đầu vào
            d_model: Dimension của model (default: 128)
            nhead: Số attention heads (default: 8)
            num_layers: Số transformer encoder layers (default: 4)
            dim_feedforward: Dimension của feedforward network (default: 512)
            dropout: Dropout rate (default: 0.1)
            forecast_days: Số ngày dự báo (1-3)
        """
        super(StockTransformer, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.forecast_days = forecast_days
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, d_model // 4)
        self.fc3 = nn.Linear(d_model // 4, forecast_days)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_size)
        Returns:
            output: (batch_size, forecast_days)
        """
        # Project input to d_model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        # Positional encoding expects (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # Lấy output từ timestep cuối cùng
        last_output = encoded[:, -1, :]  # (batch_size, d_model)
        
        # Fully connected layers
        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        output = self.fc3(x)
        
        return output


class SimpleTransformer(nn.Module):
    """Transformer đơn giản hơn để so sánh"""
    
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, forecast_days=3):
        super(SimpleTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(d_model, forecast_days)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        encoded = self.transformer_encoder(x)
        last_output = encoded[:, -1, :]
        output = self.fc(last_output)
        return output

