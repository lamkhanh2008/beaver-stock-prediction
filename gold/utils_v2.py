import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_data(master_path=None):
    if master_path is None:
        master_path = os.path.join(os.path.dirname(__file__), "gold", "data", "gold_master_history.csv")
    
    # Load master data
    df = pd.read_csv(master_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    return df

def generate_advanced_features(df):
    # 1. Price conversion and Log Returns
    df['xau_log_ret'] = np.log(df['xau_usd'] / df['xau_usd'].shift(1))
    
    # 2. Technical Indicators for XAU/USD (World Gold)
    for l in [1, 3, 5, 10]:
        df[f'xau_ret_{l}'] = df['xau_usd'].pct_change(l)
    
    # Trend Analysis
    df['xau_ma5'] = df['xau_usd'].rolling(window=5).mean()
    df['xau_ma20'] = df['xau_usd'].rolling(window=20).mean()
    df['xau_trend'] = (df['xau_ma5'] - df['xau_ma20']) / df['xau_ma20'] # Trend strength
    
    # RSI (Relative Strength Index)
    delta = df['xau_usd'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['xau_rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands Position
    df['xau_std20'] = df['xau_usd'].rolling(window=20).std()
    df['xau_bb_pos'] = (df['xau_usd'] - (df['xau_ma20'] - 2*df['xau_std20'])) / (4*df['xau_std20'] + 1e-8)
    
    # 3. SJC (Vietnam Gold) Features
    df['sjc_ret_1'] = df['sjc_price'].pct_change(1)
    df['sjc_ret_3'] = df['sjc_price'].pct_change(3)
    df['sjc_vol_5'] = df['sjc_ret_1'].rolling(5).std() # Local volatility
    
    # 4. Basis (VN vs World Gap) - THE MOST IMPORTANT SIGNAL
    df['basis'] = df['sjc_price'] - df['xau_vnd_tael']
    df['basis_ma5'] = df['basis'].rolling(window=5).mean()
    df['basis_ma20'] = df['basis'].rolling(window=20).mean()
    df['basis_zscore'] = (df['basis'] - df['basis_ma5']) / (df['basis'].rolling(window=5).std() + 1e-8)
    df['basis_accel'] = df['basis'].diff().diff() 
    df['basis_dist_ma20'] = (df['basis'] - df['basis_ma20']) / (df['basis_ma20'] + 1e-8)
    
    # 5. Macro & Inter-market
    df['dxy_ret'] = df['dxy'].pct_change(1)
    df['fed_diff'] = df['fed'].diff()
    df['yield_diff'] = df['us_10y_yield'].diff()
    df['real_yield'] = df['us_10y_yield'] - (df['xau_ret_10'] * 100)
    
    # Dynamic Correlations (The market "Mood")
    df['xau_dxy_corr'] = df['xau_log_ret'].rolling(10).corr(df['dxy'].pct_change(1)).fillna(0)
    df['xau_yield_corr'] = df['xau_log_ret'].rolling(10).corr(df['us_10y_yield'].pct_change(1)).fillna(0)
    
    # 6. Range Feature
    df['xau_range'] = df['xau_usd'].rolling(5).std() / df['xau_usd'].rolling(5).mean()
    df['basis_vol'] = df['basis'].rolling(10).std()
    
    # 6. Seasonal & Temporal
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_year_end'] = df['month'].apply(lambda x: 1 if x in [12, 1] else 0)
    
    # 7. Target Engineering (Direction of NEXT day)
    # Threshold 0.03 for more sensitivity in 1-year data
    df['target_dir'] = np.where(df['sjc_price'].shift(-1) > df['sjc_price'] + 0.03, 1, 
                               np.where(df['sjc_price'].shift(-1) < df['sjc_price'] - 0.03, -1, 0))
    df['target_ret'] = df['sjc_price'].pct_change(1).shift(-1)
    
    # Fill NaNs but don't drop rows yet to keep latest data for prediction
    df = df.fillna(0)
    return df

def get_train_test_split(df, test_size=0.15):
    # Training needs target, so we drop the last row and initial unstable rows
    df_trainable = df.iloc[30:-1]
    
    feature_cols = [
        'fed', 'us_10y_yield', 'dxy', 'xau_usd', 'xau_ret_1', 'xau_ret_5', 'xau_trend',
        'xau_bb_pos', 'xau_rsi', 'basis_zscore', 'basis_accel', 'basis_dist_ma20',
        'dxy_ret', 'fed_diff', 'yield_diff', 'real_yield', 'xau_dxy_corr', 'xau_range',
        'sjc_ret_1', 'sjc_vol_5', 'month', 'day_of_month', 'day_of_week', 'is_year_end'
    ]
        
    X_train = train = df_trainable.iloc[:int(len(df_trainable)*(1-test_size))]
    X_test = test = df_trainable.iloc[int(len(df_trainable)*(1-test_size)):]
    
    return X_train[feature_cols], X_train['target_dir'], X_test[feature_cols], X_test['target_dir'], feature_cols

