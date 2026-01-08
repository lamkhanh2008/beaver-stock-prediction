import pandas as pd
import numpy as np
import os

def load_silver_data(path=None):
    if path is None:
        path = "silver/data/silver_master_history.csv"
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date')

def generate_silver_features(df):
    """Feature Engineering nâng cao (V7) - Tập trung vào Gia tốc và Liên thị trường"""
    
    # 1. Biến động Bạc Thế giới (XAG/USD)
    df['ag_ret_1'] = df['silver_usd'].pct_change(1)
    df['ag_ret_5'] = df['silver_usd'].pct_change(5)
    
    # RSI & Momentum
    delta = df['silver_usd'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['ag_rsi'] = 100 - (100 / (1 + rs))
    
    # Rate of Change (Gia tốc)
    df['ag_roc'] = (df['silver_usd'] - df['silver_usd'].shift(5)) / (df['silver_usd'].shift(5) + 1e-8)
    
    # 2. Liên thị trường (Inter-market)
    # Tương quan Vàng - Bạc (Gold leading indicator)
    df['gold_ret_1'] = df['gold_usd'].pct_change(1)
    df['gold_ret_5'] = df['gold_usd'].pct_change(5)
    
    # Gold-Silver Ratio (GSR) acceleration
    df['gsr_ma10'] = df['gsr'].rolling(10).mean()
    df['gsr_dist_ma10'] = (df['gsr'] - df['gsr_ma10']) / df['gsr_ma10']
    
    # 3. Macro Acceleration
    df['dxy_ret'] = df['dxy'].pct_change(1)
    df['yield_ret'] = df['us_10y_yield'].pct_change(1)
    
    # 4. Vietnam Silver specific
    df['vn_buy_ret_1'] = df['silver_vn_buy'].pct_change(1)
    df['vn_sell_ret_1'] = df['silver_vn_sell'].pct_change(1)
    df['vn_spread'] = (df['silver_vn_sell'] - df['silver_vn_buy']) / (df['silver_vn_sell'] + 1e-8)
    
    # 5. Volatility & Mean Reversion
    df['ag_vol_10'] = df['ag_ret_1'].rolling(10).std()
    df['ag_ma20'] = df['silver_usd'].rolling(20).mean()
    df['dist_ma20'] = (df['silver_usd'] - df['ag_ma20']) / (df['ag_ma20'] + 1e-8)
    
    # 6. Target (Dự báo Giá Bán ngày mai)
    threshold = 0.015 # Ngưỡng 15k/lượng
    df['target_dir'] = np.where(df['silver_vn_sell'].shift(-1) > df['silver_vn_sell'] + threshold, 1,
                               np.where(df['silver_vn_sell'].shift(-1) < df['silver_vn_sell'] - threshold, -1, 0))
    
    df = df.fillna(0)
    return df

def get_train_test_split(df, test_size=0.15):
    # Loại bỏ 30 ngày đầu để các chỉ báo rolling ổn định
    df_trainable = df.iloc[30:-1]
    
    # Danh sách đặc trưng V7 mở rộng
    feature_cols = [
        'ag_ret_1', 'ag_ret_5', 'ag_rsi', 'ag_roc', 'ag_vol_10',
        'gold_ret_1', 'gold_ret_5', 'gsr_dist_ma10',
        'dxy_ret', 'yield_ret',
        'vn_buy_ret_1', 'vn_sell_ret_1', 'vn_spread'
    ]
    
    split_idx = int(len(df_trainable) * (1 - test_size))
    train = df_trainable.iloc[:split_idx]
    test = df_trainable.iloc[split_idx:]
    
    return train[feature_cols], train['target_dir'], test[feature_cols], test['target_dir'], feature_cols
    
    split_idx = int(len(df_trainable) * (1 - test_size))
    train = df_trainable.iloc[:split_idx]
    test = df_trainable.iloc[split_idx:]
    
    return train[feature_cols], train['target_dir'], test[feature_cols], test['target_dir'], feature_cols

