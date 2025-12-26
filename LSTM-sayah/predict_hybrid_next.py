import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from train_hybrid import CNNLSTMAttention, add_indicators, load_model_weights, merge_csvs


def detect_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_processed_df(df, min_rows):
    frames = []
    for sym, group in df.groupby("Symbol"):
        group_sorted = group.sort_values("Date")
        if len(group_sorted) < min_rows:
            continue
        processed = add_indicators(group_sorted)
        if processed.empty:
            continue
        processed["Symbol"] = sym
        processed["Date"] = group_sorted.loc[processed.index, "Date"].values
        frames.append(processed)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def fit_scalers(processed_df):
    feature_cols = [c for c in processed_df.columns if c not in ("Symbol", "Date")]
    f_scaler = StandardScaler()
    f_scaler.fit(processed_df[feature_cols])
    t_scaler = MinMaxScaler()
    t_scaler.fit(processed_df[["Close"]])
    return f_scaler, t_scaler, feature_cols


def build_train_df(df, train_ratio):
    train_frames = []
    for _, group in df.groupby("Symbol"):
        group = group.sort_values("Date")
        split_idx = int(len(group) * train_ratio)
        train_frames.append(group.iloc[:split_idx])
    return pd.concat(train_frames) if train_frames else None


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

    train_df = build_train_df(df, args.train_ratio) if args.scaler_source == "train" else df
    if train_df is None or train_df.empty:
        print("Error: Not enough data to fit scalers.")
        return

    processed_train = build_processed_df(train_df, args.lookback + args.forecast_days)
    if processed_train is None or processed_train.empty:
        print("Error: Not enough data to build scalers.")
        return

    f_scaler, t_scaler, feature_cols = fit_scalers(processed_train)

    processed_all = build_processed_df(df, args.lookback)
    if processed_all is None or processed_all.empty:
        print("Error: Not enough data to build prediction inputs.")
        return

    processed_all[feature_cols] = f_scaler.transform(processed_all[feature_cols])

    symbols = []
    last_dates = []
    inputs = []
    skipped = []
    for sym, group in processed_all.groupby("Symbol"):
        group_sorted = group.sort_values("Date")
        if len(group_sorted) < args.lookback:
            skipped.append(sym)
            continue
        seq = group_sorted[feature_cols].values[-args.lookback:]
        inputs.append(seq)
        symbols.append(sym)
        last_dates.append(group_sorted["Date"].iloc[-1])

    if not inputs:
        print("Error: No symbols have enough data for prediction.")
        return

    x = torch.from_numpy(np.array(inputs, dtype=np.float32))
    device = detect_device()
    model = CNNLSTMAttention(
        input_size=x.shape[2],
        hidden_size=args.hidden_size,
        forecast_days=args.forecast_days,
    ).to(device)

    loaded = load_model_weights(model, args.load_path, device)
    if not loaded:
        return

    model.eval()
    with torch.no_grad():
        preds = model(x.to(device)).cpu().numpy()

    preds_2d = preds.reshape(-1, 1)
    pred_prices = t_scaler.inverse_transform(preds_2d).reshape(preds.shape)

    out = pd.DataFrame(
        {
            "Symbol": symbols,
            "LastDate": pd.to_datetime(last_dates).strftime("%Y-%m-%d"),
        }
    )
    if args.forecast_days == 1:
        out["Pred_Close"] = pred_prices[:, 0]
    else:
        for i in range(args.forecast_days):
            out[f"Pred_Day_{i+1}"] = pred_prices[:, i]

    out = out.sort_values("Symbol")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved predictions to {args.output}")
    print(f"Predicted symbols: {len(out)}")
    if skipped:
        print(f"Skipped symbols (not enough data): {len(skipped)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="../data/all_symbols.csv")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--forecast-days", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--load-path", default="./checkpoints/best_hybrid.pt")
    parser.add_argument("--output", default="./predictions/next_day_predictions.csv")
    parser.add_argument("--scaler-source", choices=["train", "all"], default="train")
    main(parser.parse_args())
