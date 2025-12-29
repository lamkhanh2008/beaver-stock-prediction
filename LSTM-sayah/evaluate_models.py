import argparse
import os
import numpy as np
import pandas as pd


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


def build_symbol_index(df):
    symbol_index = {}
    for sym, group in df.groupby("Symbol"):
        group_sorted = group.sort_values("Date")
        dates = pd.to_datetime(group_sorted["Date"]).values
        closes = group_sorted["Close"].astype(float).values
        symbol_index[sym] = {
            "dates": dates,
            "date_index": pd.Index(dates),
            "closes": closes,
        }
    return symbol_index


def parse_pred_columns(df):
    if "Pred_Close" in df.columns:
        return {1: "Pred_Close"}
    pred_cols = [c for c in df.columns if c.startswith("Pred_Day_")]
    if not pred_cols:
        return {}
    mapping = {}
    for col in pred_cols:
        try:
            day = int(col.split("_")[-1])
        except ValueError:
            continue
        mapping[day] = col
    return dict(sorted(mapping.items()))


def evaluate_file(pred_path, symbol_index):
    df = pd.read_csv(pred_path)
    required_cols = {"Symbol", "LastDate"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns in {pred_path}: {missing}")

    pred_cols = parse_pred_columns(df)
    if not pred_cols:
        raise ValueError(f"No prediction columns found in {pred_path}")

    df["LastDate"] = pd.to_datetime(df["LastDate"])

    records = []
    for _, row in df.iterrows():
        sym = row["Symbol"]
        last_date = row["LastDate"]
        if sym not in symbol_index:
            continue
        sym_info = symbol_index[sym]
        idx = sym_info["date_index"].get_indexer([np.datetime64(last_date)])[0]
        if idx == -1:
            continue
        last_close = sym_info["closes"][idx]
        for horizon, col in pred_cols.items():
            target_idx = idx + horizon
            if target_idx >= len(sym_info["closes"]):
                continue
            actual = sym_info["closes"][target_idx]
            target_date = sym_info["dates"][target_idx]
            pred = float(row[col])
            records.append(
                {
                    "Symbol": sym,
                    "LastDate": last_date,
                    "TargetDate": pd.to_datetime(target_date),
                    "Horizon": horizon,
                    "Pred": pred,
                    "Actual": actual,
                    "LastClose": last_close,
                    "NaivePred": last_close,
                }
            )

    return pd.DataFrame(records)


def summarize_records(records):
    if records.empty:
        return pd.DataFrame()
    summaries = []
    for horizon, group in records.groupby("Horizon"):
        metrics = compute_metrics(group["Actual"], group["Pred"])
        naive_metrics = compute_metrics(group["Actual"], group["NaivePred"])
        direction_acc = np.nan
        if horizon == 1:
            direction_acc = np.mean(
                np.sign(group["Pred"] - group["LastClose"])
                == np.sign(group["Actual"] - group["LastClose"])
            ) * 100
        summaries.append(
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
    return pd.DataFrame(summaries)


def summarize_per_symbol(records):
    if records.empty:
        return pd.DataFrame()
    rows = []
    for (sym, horizon), group in records.groupby(["Symbol", "Horizon"]):
        metrics = compute_metrics(group["Actual"], group["Pred"])
        direction_acc = np.nan
        if horizon == 1:
            direction_acc = np.mean(
                np.sign(group["Pred"] - group["LastClose"])
                == np.sign(group["Actual"] - group["LastClose"])
            ) * 100
        rows.append(
            {
                "Symbol": sym,
                "Horizon": horizon,
                "n": metrics["n"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "smape": metrics["smape"],
                "direction_acc": direction_acc,
            }
        )
    return pd.DataFrame(rows)


def main(args):
    if not os.path.exists(args.data_file):
        print(f"Error: data file not found: {args.data_file}")
        return

    actual_df = pd.read_csv(args.data_file)
    if not {"Date", "Close", "Symbol"}.issubset(actual_df.columns):
        print("Error: data file missing required columns (Date, Close, Symbol).")
        return
    actual_df["Date"] = pd.to_datetime(actual_df["Date"])
    symbol_index = build_symbol_index(actual_df)

    all_summary = []
    all_per_symbol = []
    all_rows = []

    for pred_path in args.pred_files:
        if not os.path.exists(pred_path):
            print(f"Warning: prediction file not found: {pred_path}")
            continue
        try:
            records = evaluate_file(pred_path, symbol_index)
        except ValueError as exc:
            print(f"Warning: {exc}")
            continue

        model_name = os.path.splitext(os.path.basename(pred_path))[0]
        records["Model"] = model_name

        summary = summarize_records(records)
        if not summary.empty:
            summary["Model"] = model_name
            all_summary.append(summary)

        per_symbol = summarize_per_symbol(records)
        if not per_symbol.empty:
            per_symbol["Model"] = model_name
            all_per_symbol.append(per_symbol)

        all_rows.append(records)

    if not all_summary:
        print("No valid prediction files to evaluate.")
        return

    summary_df = pd.concat(all_summary, ignore_index=True)
    per_symbol_df = pd.concat(all_per_symbol, ignore_index=True) if all_per_symbol else pd.DataFrame()
    rows_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    summary_df = summary_df[
        [
            "Model",
            "horizon",
            "n",
            "mae",
            "rmse",
            "mape",
            "smape",
            "direction_acc",
            "naive_mae",
            "naive_rmse",
        ]
    ].sort_values(["horizon", "mae"])

    print("\nModel Summary:")
    print(summary_df.to_string(index=False))

    if args.out_summary:
        out_dir = os.path.dirname(args.out_summary)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        summary_df.to_csv(args.out_summary, index=False)
        print(f"\nSaved summary to {args.out_summary}")

    if args.out_per_symbol and not per_symbol_df.empty:
        out_dir = os.path.dirname(args.out_per_symbol)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        per_symbol_df.to_csv(args.out_per_symbol, index=False)
        print(f"Saved per-symbol metrics to {args.out_per_symbol}")

    if args.out_rows and not rows_df.empty:
        out_dir = os.path.dirname(args.out_rows)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        rows_df.to_csv(args.out_rows, index=False)
        print(f"Saved detailed rows to {args.out_rows}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="../data/all_symbols.csv")
    parser.add_argument("--pred-files", nargs="+", required=True)
    parser.add_argument("--out-summary", default="./eval/model_summary.csv")
    parser.add_argument("--out-per-symbol", default="./eval/per_symbol_metrics.csv")
    parser.add_argument("--out-rows", default=None)
    main(parser.parse_args())
