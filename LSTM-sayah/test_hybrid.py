import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train_hybrid import (
    CNNLSTMAttention,
    eval_epoch_loss,
    load_model_weights,
    merge_csvs,
    prepare_sequences_multi,
    predict_batches,
)


def detect_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_splits(df, train_ratio):
    train_frames, test_frames = [], []
    for _, group in df.groupby("Symbol"):
        group = group.sort_values("Date")
        split_idx = int(len(group) * train_ratio)
        train_frames.append(group.iloc[:split_idx])
        test_frames.append(group.iloc[split_idx:])
    train_df = pd.concat(train_frames) if train_frames else None
    test_df = pd.concat(test_frames) if test_frames else None
    return train_df, test_df


def main(args):
    if not os.path.exists(args.data_file):
        print(f"Merging data from {args.data_dir}...")
        merge_csvs(args.data_dir, args.data_file)

    if not os.path.exists(args.data_file):
        print("Error: No data found.")
        return

    if not os.path.exists(args.load_path):
        print(f"Error: load path not found: {args.load_path}")
        return

    df = pd.read_csv(args.data_file)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df["Date"] = pd.to_datetime(df[date_col])

    train_df, test_df = build_splits(df, args.train_ratio)
    if train_df is None or test_df is None or train_df.empty or test_df.empty:
        print("Error: Not enough data for train/test split.")
        return

    X_train, y_train, f_scaler, t_scaler = prepare_sequences_multi(
        train_df, args.lookback, args.forecast_days
    )
    X_test, y_test, _, _ = prepare_sequences_multi(
        test_df, args.lookback, args.forecast_days, f_scaler, t_scaler
    )

    if X_train is None or X_test is None or len(X_test) == 0:
        print("Error: Not enough data after sequence creation.")
        return

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = detect_device()
    model = CNNLSTMAttention(
        input_size=X_train.shape[2],
        hidden_size=args.hidden_size,
        forecast_days=args.forecast_days,
    ).to(device)

    loaded = load_model_weights(model, args.load_path, device)
    if not loaded:
        return

    criterion = nn.HuberLoss()

    print("--- Test Hybrid Model ---")
    print(f"Device: {device}")
    print(f"Test samples: {len(X_test)}")

    test_loss = eval_epoch_loss(model, test_loader, criterion, device)
    preds = predict_batches(model, test_loader, device)
    if preds.size == 0:
        print("Final Test Loss: N/A (no test predictions)")
        return

    actual_prices = t_scaler.inverse_transform(y_test)
    pred_prices = t_scaler.inverse_transform(preds)
    mae = np.mean(np.abs(actual_prices - pred_prices))
    rmse = np.sqrt(np.mean((actual_prices - pred_prices) ** 2))
    print(f"Final Test Loss (Huber): {test_loss:.6f}")
    print(f"Final Test MAE: {mae:.2f}")
    print(f"Final Test RMSE: {rmse:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="../data/all_symbols.csv")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--forecast-days", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--load-path", default="./checkpoints/best_hybrid.pt")
    main(parser.parse_args())
