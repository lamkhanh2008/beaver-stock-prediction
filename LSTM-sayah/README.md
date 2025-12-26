# LSTM-sayah

Kaggle-style LSTM for next-day stock price prediction using `data/all_symbols.csv`.

## Usage
```bash
python LSTM-sayah/train_lstm.py
```

Optional arguments:
```bash
python LSTM-sayah/train_lstm.py --lookback 60 --epochs 50 --train-ratio 0.8
```

The script will:
- Read `data/all_symbols.csv` (and merge from `data/` if missing)
- Train a single LSTM on all symbols (Close price only)
- Print test metrics and next-day predictions per symbol
