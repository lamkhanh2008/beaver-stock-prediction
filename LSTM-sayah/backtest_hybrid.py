import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from train_hybrid import CNNLSTMAttention, add_indicators, load_model_weights, merge_csvs


def detect_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_mape(actual, pred):
    actual = np.array(actual, dtype=float)
    pred = np.array(pred, dtype=float)
    mask = actual != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100


def safe_smape(actual, pred):
    actual = np.array(actual, dtype=float)
    pred = np.array(pred, dtype=float)
    denom = (np.abs(actual) + np.abs(pred)) / 2.0
    mask = denom != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs(actual[mask] - pred[mask]) / denom[mask]) * 100


def compute_metrics(actual, pred):
    actual = np.array(actual, dtype=float)
    pred = np.array(pred, dtype=float)
    if len(actual) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "smape": np.nan,
        }
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    return {
        "n": len(actual),
        "mae": mae,
        "rmse": rmse,
        "mape": safe_mape(actual, pred),
        "smape": safe_smape(actual, pred),
    }


def build_processed_groups(df, train_ratio):
    groups = {}
    train_last_dates = {}
    for sym, group in df.groupby("Symbol"):
        group_sorted = group.sort_values("Date")
        if group_sorted.empty:
            continue
        split_idx = int(len(group_sorted) * train_ratio)
        train_last_date = None
        if split_idx > 0:
            train_last_date = pd.to_datetime(group_sorted.iloc[split_idx - 1]["Date"])
        processed = add_indicators(group_sorted)
        if processed is None or processed.empty:
            continue
        processed = processed.copy()
        processed["Date"] = pd.to_datetime(group_sorted.loc[processed.index, "Date"]).values
        processed["Symbol"] = sym
        processed["Close_raw"] = group_sorted.loc[processed.index, "Close"].values
        processed = processed.reset_index(drop=True)
        groups[sym] = processed
        train_last_dates[sym] = train_last_date
    return groups, train_last_dates


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

    groups, train_last_dates = build_processed_groups(df, args.train_ratio)
    if not groups:
        print("Error: Not enough data after preprocessing.")
        return

    sample_group = next(iter(groups.values()))
    feature_cols = [c for c in sample_group.columns if c not in ("Symbol", "Date", "Close_raw")]

    train_frames = []
    for sym, group in groups.items():
        last_date = train_last_dates.get(sym)
        if last_date is None:
            continue
        train_frames.append(group[group["Date"] <= last_date])

    if not train_frames:
        print("Error: Not enough training data to fit scalers.")
        return

    train_df = pd.concat(train_frames, ignore_index=True)
    f_scaler = StandardScaler()
    f_scaler.fit(train_df[feature_cols])
    t_scaler = MinMaxScaler()
    t_scaler.fit(train_df[["Close_raw"]])

    sequences = []
    actuals = []
    meta = []
    date_arrays = []

    for sym, group in groups.items():
        last_date = train_last_dates.get(sym)
        if last_date is None:
            continue
        group_sorted = group.sort_values("Date").reset_index(drop=True)
        group_sorted[feature_cols] = f_scaler.transform(group_sorted[feature_cols])

        feat_array = group_sorted[feature_cols].values
        close_array = group_sorted["Close_raw"].astype(float).values
        dates = pd.to_datetime(group_sorted["Date"]).values

        for i in range(args.lookback, len(feat_array) - args.forecast_days + 1):
            target_date = pd.to_datetime(dates[i])
            if target_date <= last_date:
                continue
            seq = feat_array[i - args.lookback : i]
            target_close = close_array[i : i + args.forecast_days]
            target_dates = dates[i : i + args.forecast_days]
            sequences.append(seq)
            actuals.append(target_close)
            date_arrays.append(target_dates)
            meta.append(
                {
                    "Symbol": sym,
                    "LastDate": dates[i - 1],
                    "LastClose": close_array[i - 1],
                }
            )

    if not sequences:
        print("Error: No test sequences generated.")
        return

    X = torch.from_numpy(np.array(sequences, dtype=np.float32))
    y = np.array(actuals, dtype=np.float32)

    device = detect_device()
    model = CNNLSTMAttention(
        input_size=X.shape[2],
        hidden_size=args.hidden_size,
        forecast_days=args.forecast_days,
    ).to(device)

    loaded = load_model_weights(model, args.load_path, device)
    if not loaded:
        return

    model.eval()
    loader = DataLoader(TensorDataset(X), batch_size=args.batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (b_x,) in loader:
            pred = model(b_x.to(device)).cpu().numpy()
            preds.append(pred)
    preds = np.concatenate(preds, axis=0)

    preds_2d = preds.reshape(-1, 1)
    preds_unscaled = t_scaler.inverse_transform(preds_2d).reshape(preds.shape)

    summary_rows = []
    for horizon in range(1, args.forecast_days + 1):
        actual_h = y[:, horizon - 1]
        pred_h = preds_unscaled[:, horizon - 1]
        metrics = compute_metrics(actual_h, pred_h)
        naive_pred = np.array([m["LastClose"] for m in meta], dtype=float)
        naive_metrics = compute_metrics(actual_h, naive_pred)
        direction_acc = np.nan
        if horizon == 1:
            direction_acc = (
                np.mean(np.sign(pred_h - naive_pred) == np.sign(actual_h - naive_pred)) * 100
            )
        summary_rows.append(
            {
                "horizon": horizon,
                "n": metrics["n"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "smape": metrics["smape"],
                "direction_acc": direction_acc,
                "naive_mae": naive_metrics["mae"],
                "naive_rmse": naive_metrics["rmse"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print("\nBacktest Summary:")
    print(summary_df.to_string(index=False))

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    out_rows = []
    for i, m in enumerate(meta):
        row = {
            "Symbol": m["Symbol"],
            "LastDate": pd.to_datetime(m["LastDate"]).strftime("%Y-%m-%d"),
            "LastClose": m["LastClose"],
        }
        for h in range(args.forecast_days):
            row[f"TargetDate_Day_{h+1}"] = pd.to_datetime(date_arrays[i][h]).strftime(
                "%Y-%m-%d"
            )
            row[f"Actual_Day_{h+1}"] = y[i, h]
            row[f"Pred_Day_{h+1}"] = preds_unscaled[i, h]
        if args.forecast_days >= 1:
            row["Direction_Correct"] = int(
                np.sign(preds_unscaled[i, 0] - m["LastClose"])
                == np.sign(y[i, 0] - m["LastClose"])
            )
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved backtest rows to {args.output}")

    if args.out_summary:
        summary_dir = os.path.dirname(args.out_summary)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        summary_df.to_csv(args.out_summary, index=False)
        print(f"Saved summary to {args.out_summary}")

    if args.out_per_symbol:
        per_rows = []
        for sym, group in out_df.groupby("Symbol"):
            for h in range(args.forecast_days):
                actual_h = group[f"Actual_Day_{h+1}"].values
                pred_h = group[f"Pred_Day_{h+1}"].values
                metrics = compute_metrics(actual_h, pred_h)
                direction_acc = np.nan
                if h == 0:
                    direction_acc = np.mean(group["Direction_Correct"]) * 100
                per_rows.append(
                    {
                        "Symbol": sym,
                        "Horizon": h + 1,
                        "n": metrics["n"],
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "mape": metrics["mape"],
                        "smape": metrics["smape"],
                        "direction_acc": direction_acc,
                    }
                )
        per_df = pd.DataFrame(per_rows)
        per_dir = os.path.dirname(args.out_per_symbol)
        if per_dir:
            os.makedirs(per_dir, exist_ok=True)
        per_df.to_csv(args.out_per_symbol, index=False)
        print(f"Saved per-symbol metrics to {args.out_per_symbol}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="../data/all_symbols.csv")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--forecast-days", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--load-path", default="./checkpoints/best_hybrid.pt")
    parser.add_argument("--output", default="./backtest/backtest_rows.csv")
    parser.add_argument("--out-summary", default="./backtest/backtest_summary.csv")
    parser.add_argument("--out-per-symbol", default="./backtest/backtest_per_symbol.csv")
    main(parser.parse_args())
