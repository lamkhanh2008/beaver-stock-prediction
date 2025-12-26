"""
Hybrid CNN-LSTM-Attention Model cho dự báo giá chứng khoán
Kết hợp CNN để trích xuất đặc trưng cục bộ, LSTM cho chuỗi thời gian 
và Attention để tập trung vào các thời điểm quan trọng.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, n_heads=8, dropout=0.2, forecast_days=1):
        super(CNNLSTMAttention, self).__init__()
        
        # 1. CNN Layer: Trích xuất local patterns từ các features
        # Dùng Conv1d để quét qua các timestep
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2) if False else nn.Identity() # Tùy chọn pool nếu seq_len dài
        
        # 2. LSTM Layer: Học các phụ thuộc thời gian (temporal dependencies)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Do dùng bidirectional, output size của LSTM sẽ là hidden_size * 2
        lstm_output_dim = hidden_size * 2
        
        # 3. Attention Mechanism: Multi-head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 4. Fully Connected Layers
        self.fc1 = nn.Linear(lstm_output_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, forecast_days)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        
    def forward(self, x):
        """
        Input x: (batch_size, seq_len, input_size)
        """
        # CNN expects (batch_size, channels, seq_len)
        x_cnn = x.transpose(1, 2)
        x_cnn = F.relu(self.conv1(x_cnn))
        x_cnn = F.relu(self.conv2(x_cnn))
        
        # Quay lại (batch_size, seq_len, hidden_size) cho LSTM
        x_lstm_in = x_cnn.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x_lstm_in) # (batch_size, seq_len, lstm_output_dim)
        
        # Attention
        # Dùng timestep cuối cùng làm query để tìm các phần quan trọng trong quá khứ
        query = lstm_out[:, -1:, :] # (batch_size, 1, lstm_output_dim)
        attn_out, _ = self.attention(query, lstm_out, lstm_out)
        attn_out = attn_out.squeeze(1) # (batch_size, lstm_output_dim)
        
        # Residual Connection & Layer Norm (tùy chọn)
        # res = lstm_out[:, -1, :]
        # attn_out = self.layer_norm(attn_out + res)
        
        # FC layers
        x = F.relu(self.fc1(attn_out))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.fc3(x)
        
        return out

class GRUAttention(nn.Module):
    """Sử dụng GRU thay cho LSTM (thường nhanh hơn và đôi khi chính xác hơn)"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, n_heads=8, dropout=0.2, forecast_days=1):
        super(GRUAttention, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_size * 2
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, forecast_days)
        )
        
    def forward(self, x):
        out, _ = self.gru(x)
        query = out[:, -1:, :]
        attn_out, _ = self.attention(query, out, out)
        attn_out = attn_out.squeeze(1)
        return self.fc(attn_out)

