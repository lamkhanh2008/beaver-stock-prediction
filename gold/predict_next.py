import os
import argparse
import joblib
import pandas as pd
import numpy as np

from utils import load_gold_csv, generate_features, select_feature_target

BASE_DIR = os.path.dirname(__file__)
DEFAULT_DATA_FILES = [
    os.path.join(BASE_DIR, "data", "sjc.csv"),
    os.path.join(BASE_DIR, "data", "nhantron.csv"),
]
ALT_DEFAULT_DATA_FILES = [
    os.path.join(BASE_DIR, "gold", "data", "sjc.csv"),
    os.path.join(BASE_DIR, "gold", "data", "nhantron.csv"),
]


def resolve_data_files(paths):
    existing = [p for p in paths if p and os.path.exists(p)]
    if existing:
        return existing
    alt = [p for p in ALT_DEFAULT_DATA_FILES if os.path.exists(p)]
    return alt


def main(args):
    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        return
    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    target_type = bundle.get("target_type", "price")

    df = load_gold_csv(args.data_files)
    if df.empty:
        print("No data loaded.")
        return

    df_feat = generate_features(df, price_col="sell")
    if df_feat.empty:
        print("No data after feature generation.")
        return

    target_col = "target_return_t1" if target_type == "return" else "target_price_t1"
    X, y, base_price = select_feature_target(df_feat, target_col=target_col)
    X = X[feature_names]
    last_row = X.iloc[[-1]]
    pred = model.predict(last_row)[0]
    last_dt = df_feat.iloc[-1]["datetime"]
    last_price = df_feat.iloc[-1]["price"]
    if target_type == "return":
        pred_price = last_price * (1 + pred)
    else:
        pred_price = pred

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df = pd.DataFrame(
        [
            {
                "last_datetime": last_dt,
                "last_price": last_price,
                "pred_price_t1": pred_price,
                "pred_return_t1": pred if target_type == "return" else np.nan,
            }
        ]
    )
    out_df.to_csv(args.output, index=False)
    print(f"Saved next-day prediction to {args.output}")
    print(out_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.path.join(BASE_DIR, "models", "gbr_model.joblib"))
    parser.add_argument("--data-files", nargs="+", default=None)
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "predictions", "next_day.csv"))
    args = parser.parse_args()
    if args.data_files is None:
        args.data_files = resolve_data_files(DEFAULT_DATA_FILES)
    else:
        args.data_files = resolve_data_files(args.data_files)
    if not args.data_files:
        print("No data files found. Please check paths.")
        exit(1)
    print("Using data files:", args.data_files)
    main(args)
