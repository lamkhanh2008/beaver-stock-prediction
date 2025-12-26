import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def merge_csvs(data_dir, output_file):
    """Merge *_data.csv files into one CSV with a Symbol column."""
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_data.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No *_data.csv files found in {data_dir}")

    frames = []
    for file in csv_files:
        sym = file.replace('_data.csv', '')
        path = os.path.join(data_dir, file)
        df = pd.read_csv(path)
        if df.empty:
            continue
        date_col = df.columns[0]
        df = df.rename(columns={date_col: 'Date'})
        df['Symbol'] = sym
        frames.append(df)

    if not frames:
        raise ValueError("No valid data to merge.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_file, index=False)
    return output_file


def load_data(data_file, data_dir=None, start_date=None, end_date=None):
    """Load combined CSV; create it from data_dir if missing."""
    if not os.path.exists(data_file):
        if not data_dir:
            raise FileNotFoundError(f"Missing data file: {data_file}")
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        merge_csvs(data_dir, data_file)

    df = pd.read_csv(data_file)
    if df.empty:
        raise ValueError("Data file is empty.")

    if 'Symbol' not in df.columns:
        raise ValueError("Data file must include a 'Symbol' column.")

    date_col = 'Date' if 'Date' in df.columns else df.columns[0]
    df = df.rename(columns={date_col: 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])

    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]

    return df.sort_values(['Symbol', 'Date'])


def build_datasets(df, lookback=60, forecast_days=1, train_ratio=0.8, val_ratio=0.0):
    """Build sequences per symbol with scaler fit on train split only."""
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    val_symbols, test_symbols = [], []
    scalers = {}
    last_sequences = {}
    stats = defaultdict(int)

    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if val_ratio < 0 or (train_ratio + val_ratio) >= 1:
        raise ValueError("val_ratio must be >=0 and train_ratio+val_ratio < 1.")

    for sym, group in df.groupby('Symbol', sort=False):
        group = group.sort_values('Date')
        close_values = group['Close'].astype(float).values.reshape(-1, 1)
        if len(close_values) < lookback + forecast_days + 1:
            stats['skipped_short'] += 1
            continue

        train_end = int(len(close_values) * train_ratio)
        val_end = int(len(close_values) * (train_ratio + val_ratio))

        scaler = MinMaxScaler()
        scaler.fit(close_values[:train_end])
        scaled_all = scaler.transform(close_values)

        sym_train, sym_val, sym_test = 0, 0, 0
        for i in range(lookback, len(scaled_all) - forecast_days + 1):
            x = scaled_all[i - lookback:i]
            y = scaled_all[i:i + forecast_days].flatten()
            if i < train_end:
                X_train.append(x)
                y_train.append(y)
                sym_train += 1
            elif i < val_end:
                X_val.append(x)
                y_val.append(y)
                val_symbols.append(sym)
                sym_val += 1
            else:
                X_test.append(x)
                y_test.append(y)
                test_symbols.append(sym)
                sym_test += 1

        if sym_train == 0 or (len(X_test) == 0 and len(X_val) == 0):
            stats['skipped_no_split'] += 1
            continue

        scalers[sym] = scaler
        last_sequences[sym] = scaled_all[-lookback:]
        stats['used'] += 1

    if not X_train:
        raise ValueError("No training sequences were created.")

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    meta = {
        'stats': dict(stats),
        'val_symbols': val_symbols,
        'test_symbols': test_symbols,
        'scalers': scalers,
        'last_sequences': last_sequences,
    }
    return X_train, y_train, X_val, y_val, X_test, y_test, meta


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=3, dropout=0.2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def evaluate(model, loader, device):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x).cpu().numpy()
            preds.append(outputs)
            actuals.append(batch_y.numpy())
    if not preds:
        return None, None
    return np.vstack(preds), np.vstack(actuals)


def inverse_by_symbol(values, symbols, scalers):
    """Inverse transform a list of scaled values by symbol."""
    restored = []
    for val, sym in zip(values, symbols):
        scaler = scalers.get(sym)
        if scaler is None:
            restored.append(np.nan)
            continue
        restored.append(scaler.inverse_transform(val.reshape(1, -1))[0][0])
    return np.array(restored, dtype=np.float32)


def train(args):
    df = load_data(
        data_file=args.data_file,
        data_dir=args.data_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    X_train, y_train, X_val, y_val, X_test, y_test, meta = build_datasets(
        df,
        lookback=args.lookback,
        forecast_days=args.forecast_days,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_size=args.forecast_days,
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = None
    if len(X_val) > 0:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            batch_size=args.batch_size,
            shuffle=False,
        )
    test_loader = None
    if len(X_test) > 0:
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
            batch_size=args.batch_size,
            shuffle=False,
        )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Device: {device}")
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Symbols used: {meta['stats'].get('used', 0)}")
    print(f"Lookback: {args.lookback}, Forecast: {args.forecast_days}")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = None
        if val_loader:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_x)
                    val_total += criterion(outputs, batch_y).item()
            val_loss = val_total / len(val_loader)

        if val_loss is None:
            print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {avg_train_loss:.6f}")
        else:
            print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

    if test_loader:
        preds_scaled, actuals_scaled = evaluate(model, test_loader, device)
        if preds_scaled is not None:
            preds_scaled = preds_scaled.reshape(-1, args.forecast_days)
            actuals_scaled = actuals_scaled.reshape(-1, args.forecast_days)
            test_symbols = meta['test_symbols']
            pred_prices = inverse_by_symbol(preds_scaled[:, 0], test_symbols, meta['scalers'])
            actual_prices = inverse_by_symbol(actuals_scaled[:, 0], test_symbols, meta['scalers'])
            mse = np.mean((pred_prices - actual_prices) ** 2)
            mae = np.mean(np.abs(pred_prices - actual_prices))
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual_prices - pred_prices) / (actual_prices + 1e-8))) * 100
            print(f"\nTest MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

    # Next-day prediction per symbol
    print("\nNext day prediction:")
    model.eval()
    for sym, seq in meta['last_sequences'].items():
        seq_tensor = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_scaled = model(seq_tensor).cpu().numpy().flatten()
        scaler = meta['scalers'].get(sym)
        if scaler is None:
            continue
        pred_price = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        print(f"- {sym}: {pred_price:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="LSTM stock prediction (Kaggle-style).")
    parser.add_argument("--data-file", default="data/all_symbols.csv")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--forecast-days", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=50)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
