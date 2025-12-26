"""
LSTM Model cho dự báo giá cổ phiếu
Hỗ trợ nhiều input features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StockLSTM(nn.Module):
    """LSTM Model với nhiều layers cho dự báo giá cổ phiếu"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2, forecast_days=3):
        """
        Args:
            input_size: Số lượng features đầu vào
            hidden_size: Kích thước hidden layer (default: 128)
            num_layers: Số layers LSTM (default: 3)
            dropout: Dropout rate (default: 0.2)
            forecast_days: Số ngày dự báo (1-3)
        """
        super(StockLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_days = forecast_days
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism (optional, để cải thiện performance)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, forecast_days)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_size)
        Returns:
            output: (batch_size, forecast_days)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Lấy output từ layer cuối cùng
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        # Lấy output tại timestep cuối cùng
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply attention (optional)
        # Reshape để attention
        lstm_out_reshaped = lstm_out.unsqueeze(0)  # (1, batch_size, seq_len, hidden_size)
        # Simplified: use last output with attention weights
        attended_out, _ = self.attention(
            last_output.unsqueeze(1), 
            lstm_out, 
            lstm_out
        )
        attended_out = attended_out.squeeze(1)  # (batch_size, hidden_size)
        
        # Fully connected layers
        x = self.fc1(attended_out)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        output = self.fc3(x)
        
        return output


class SimpleLSTM(nn.Module):
    """LSTM đơn giản hơn để so sánh"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, forecast_days=3):
        super(SimpleLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, forecast_days)
        
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output

