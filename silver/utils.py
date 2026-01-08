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
    """Feature Engineering tập trung vào các tín hiệu mạnh và ổn định cho Bạc"""
    
    # 1. Biến động Bạc Thế giới (XAG/USD) - Đưa về log return để chuẩn hóa
    df['ag_ret_1'] = df['silver_usd'].pct_change(1)
    df['ag_ret_3'] = df['silver_usd'].pct_change(3)
    df['ag_ret_5'] = df['silver_usd'].pct_change(5)
    
    # RSI (Chỉ số sức mạnh tương đối)
    delta = df['silver_usd'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['ag_rsi'] = 100 - (100 / (1 + rs))
    
    # Khoảng cách so với MA20 (Mean Reversion)
    df['ma20'] = df['silver_usd'].rolling(window=20).mean()
    df['dist_ma20'] = (df['silver_usd'] - df['ma20']) / (df['ma20'] + 1e-8)
    
    # 2. Gold-Silver Ratio (GSR) - Tín hiệu quan trọng nhất cho Bạc
    # Nếu GSR quá cao, Bạc thường tăng mạnh để bắt kịp Vàng
    df['gsr_ma10'] = df['gsr'].rolling(10).mean()
    df['gsr_dist_ma10'] = (df['gsr'] - df['gsr_ma10']) / df['gsr_ma10']
    
    # 3. Macro & Corrs
    df['dxy_ret'] = df['dxy'].pct_change(1)
    # Tương quan động giữa Bạc và DXY
    df['ag_dxy_corr'] = df['ag_ret_1'].rolling(10).corr(df['dxy_ret']).fillna(0)
    
    # 4. Vietnam Silver Momentum (Mua vào & Bán ra)
    df['vn_buy_ret_1'] = df['silver_vn_buy'].pct_change(1)
    df['vn_sell_ret_1'] = df['silver_vn_sell'].pct_change(1)
    df['vn_spread'] = (df['silver_vn_sell'] - df['silver_vn_buy']) / (df['silver_vn_sell'] + 1e-8)
    
    # Trend indicators
    df['ag_sma5'] = df['silver_usd'].rolling(5).mean()
    df['ag_sma10'] = df['silver_usd'].rolling(10).mean()
    df['ag_trend'] = np.where(df['ag_sma5'] > df['ag_sma10'], 1, -1)
    
    # Volatility
    df['ag_std5'] = df['ag_ret_1'].rolling(5).std()
    
    # 5. Temporal
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # 6. Target - Dự báo hướng đi của GIÁ BÁN cho ngày mai
    threshold = 0.01 
    df['target_dir'] = np.where(df['silver_vn_sell'].shift(-1) > df['silver_vn_sell'] + threshold, 1,
                               np.where(df['silver_vn_sell'].shift(-1) < df['silver_vn_sell'] - threshold, -1, 0))
    
    df = df.fillna(0)
    return df

def get_train_test_split(df, test_size=0.15):
    df_trainable = df.iloc[30:-1]
    
    # Chọn lọc các đặc trưng bao gồm cả dữ liệu Mua/Bán và Chênh lệch (Spread)
    feature_cols = [
        'ag_ret_1', 'ag_ret_3', 'ag_rsi', 'dist_ma20',
        'gsr', 'gsr_dist_ma10', 'dxy_ret', 'ag_dxy_corr',
        'vn_buy_ret_1', 'vn_sell_ret_1', 'vn_spread', 
        'ag_trend', 'ag_std5', 'day_of_week'
    ]
    
    split_idx = int(len(df_trainable) * (1 - test_size))
    train = df_trainable.iloc[:split_idx]
    test = df_trainable.iloc[split_idx:]
    
    return train[feature_cols], train['target_dir'], test[feature_cols], test['target_dir'], feature_cols

