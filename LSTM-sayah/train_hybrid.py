import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from ta import add_all_ta_features
from tqdm import tqdm

# --- 1. Model Architecture ---
class CNNLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, n_heads=4, dropout=0.2, forecast_days=1):
        super().__init__()
        # CNN trích xuất local features
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # LSTM học chuỗi thời gian
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        # Attention tập trung vào timestep quan trọng
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=n_heads,
            batch_first=True
        )
        # Output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, forecast_days)
        )

    def forward(self, x):
        # x: (batch, seq, features) -> transpose thành (batch, features, seq) cho CNN
        x_cnn = x.transpose(1, 2)
        x_cnn = self.conv(x_cnn).transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x_cnn) # (batch, seq, hidden*2)
        
        # Attention (lấy timestep cuối làm query)
        query = lstm_out[:, -1:, :]
        attn_out, _ = self.attention(query, lstm_out, lstm_out)
        
        return self.fc(attn_out.squeeze(1))

# --- 2. Data Processing ---
def merge_csvs(data_dir, output_file):
    """Merge *_data.csv files into one CSV with a Symbol column."""
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_data.csv')]
    if not csv_files:
        return None

    frames = []
    for file in csv_files:
        sym = file.replace('_data.csv', '')
        path = os.path.join(data_dir, file)
        df = pd.read_csv(path)
        if df.empty: continue
        date_col = df.columns[0]
        df = df.rename(columns={date_col: 'Date'})
        df['Symbol'] = sym
        frames.append(df)

    if not frames: return None
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_file, index=False)
    return output_file

def add_indicators(df):
    """Thêm các chỉ báo kỹ thuật cơ bản"""
    df = df.copy()
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Thêm chỉ báo kỹ thuật
    df = add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )
    cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
            'momentum_rsi', 'trend_macd', 'trend_macd_signal', 
            'volatility_bbm', 'trend_sma_fast', 'trend_sma_slow']
    return df[cols]

def prepare_sequences_multi(df, lookback, forecast_days, f_scaler=None, t_scaler=None):
    """Tạo chuỗi dữ liệu cho nhiều mã cổ phiếu"""
    X_list, y_list = [], []
    
    # 1. Thu thập tất cả data đã có indicators
    all_processed_groups = []
    for sym, group in df.groupby('Symbol'):
        if len(group) < lookback + forecast_days: continue
        processed = add_indicators(group.sort_values('Date'))
        processed['Symbol'] = sym
        all_processed_groups.append(processed)
    
    if not all_processed_groups:
        return None, None, f_scaler, t_scaler
        
    full_df = pd.concat(all_processed_groups)
    feature_cols = [c for c in full_df.columns if c != 'Symbol']
    
    # 2. Scaling
    if f_scaler is None:
        f_scaler = StandardScaler()
        full_df[feature_cols] = f_scaler.fit_transform(full_df[feature_cols])
    else:
        full_df[feature_cols] = f_scaler.transform(full_df[feature_cols])
        
    if t_scaler is None:
        t_scaler = MinMaxScaler()
        t_scaler.fit(full_df[['Close']])
    
    # 3. Tạo sequence theo từng mã
    for sym, group in full_df.groupby('Symbol'):
        feat_array = group[feature_cols].values
        target_array = t_scaler.transform(group[['Close']])
        
        for i in range(lookback, len(feat_array) - forecast_days + 1):
            X_list.append(feat_array[i - lookback:i])
            y_list.append(target_array[i:i + forecast_days].flatten())
            
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), f_scaler, t_scaler

# --- 3. Training Logic ---
def eval_epoch_loss(model, data_loader, criterion, device):
    if data_loader is None or len(data_loader) == 0:
        return None
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for b_x, b_y in data_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            pred = model(b_x)
            loss = criterion(pred, b_y)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def predict_batches(model, data_loader, device):
    if data_loader is None or len(data_loader) == 0:
        return np.array([])
    model.eval()
    preds = []
    with torch.no_grad():
        for b_x, _ in data_loader:
            b_x = b_x.to(device)
            pred = model(b_x)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.array([])

def save_model_weights(model, save_path):
    if not save_path:
        return
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)

def train(args):
    # Load data
    if not os.path.exists(args.data_file):
        print(f"Merging data from {args.data_dir}...")
        merge_csvs(args.data_dir, args.data_file)
        
    if not os.path.exists(args.data_file):
        print("Error: No data found.")
        return

    df = pd.read_csv(args.data_file)
    date_col = 'Date' if 'Date' in df.columns else df.columns[0]
    df['Date'] = pd.to_datetime(df[date_col])
    
    # Split Train/Val/Test
    train_frames, val_frames, test_frames = [], [], []
    for sym, group in df.groupby('Symbol'):
        group = group.sort_values('Date')
        split_idx = int(len(group) * args.train_ratio)
        train_full = group.iloc[:split_idx]
        test_frames.append(group.iloc[split_idx:])

        if args.val_ratio > 0:
            val_split_idx = int(len(train_full) * (1 - args.val_ratio))
            if 0 < val_split_idx < len(train_full):
                train_frames.append(train_full.iloc[:val_split_idx])
                val_frames.append(train_full.iloc[val_split_idx:])
            else:
                train_frames.append(train_full)
        else:
            train_frames.append(train_full)
        
    train_df = pd.concat(train_frames)
    val_df = pd.concat(val_frames) if val_frames else None
    test_df = pd.concat(test_frames) if test_frames else None
    
    X_train, y_train, f_scaler, t_scaler = prepare_sequences_multi(train_df, args.lookback, args.forecast_days)
    X_val, y_val, _, _ = (None, None, f_scaler, t_scaler)
    if val_df is not None and not val_df.empty:
        X_val, y_val, _, _ = prepare_sequences_multi(val_df, args.lookback, args.forecast_days, f_scaler, t_scaler)
    X_test, y_test, _, _ = (None, None, f_scaler, t_scaler)
    if test_df is not None and not test_df.empty:
        X_test, y_test, _, _ = prepare_sequences_multi(test_df, args.lookback, args.forecast_days, f_scaler, t_scaler)
    
    if X_train is None:
        print("Error: Not enough data.")
        return

    # Dataloader
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), 
                              batch_size=args.batch_size, shuffle=True)
    val_loader = None
    if X_val is not None and len(X_val) > 0:
        val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                                batch_size=args.batch_size, shuffle=False)
    test_loader = None
    if X_test is not None and len(X_test) > 0:
        test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                                 batch_size=args.batch_size, shuffle=False)
    
    # Device Detection (Support Apple Silicon MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model = CNNLSTMAttention(
        input_size=X_train.shape[2], 
        hidden_size=args.hidden_size,
        forecast_days=args.forecast_days
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.HuberLoss()
    
    print(f"--- Training Hybrid Model ---")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Train samples: {len(X_train)}")
    if val_loader is not None:
        print(f"Val samples: {len(X_val)}")
    if test_loader is not None:
        print(f"Test samples: {len(X_test)}")

    best_metric = float("inf")
    best_epoch = 0
    best_tag = "val" if val_loader is not None else "train"
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            pred = model(b_x)
            loss = criterion(pred, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        val_loss = eval_epoch_loss(model, val_loader, criterion, device)
        metric = val_loss if val_loss is not None else train_loss
        if args.save_path and metric < best_metric:
            best_metric = metric
            best_epoch = epoch + 1
            save_model_weights(model, args.save_path)
            print(f"Saved best model to {args.save_path} (epoch {best_epoch}, {best_tag} loss {best_metric:.6f})")
        if val_loss is None:
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Evaluation
    model.eval()
    if test_loader is not None:
        test_loss = eval_epoch_loss(model, test_loader, criterion, device)
        preds = predict_batches(model, test_loader, device)
        if preds.size > 0:
            actual_prices = t_scaler.inverse_transform(y_test)
            pred_prices = t_scaler.inverse_transform(preds)
            mae = np.mean(np.abs(actual_prices - pred_prices))
            print(f"\nFinal Test Loss (Huber): {test_loss:.6f}")
            print(f"Final Test MAE: {mae:.2f}")
        else:
            print("\nFinal Test Loss: N/A (no test predictions)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="../data/all_symbols.csv")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--forecast-days", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save-path", default="./checkpoints/best_hybrid.pt")
    args = parser.parse_args()
    train(args)
