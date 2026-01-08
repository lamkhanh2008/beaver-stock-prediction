import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from io import StringIO

def get_stooq_data(symbol):
    """Lấy dữ liệu từ Stooq (1 năm qua)"""
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200 and "Date,Open" in response.text:
            df = pd.read_csv(StringIO(response.text))
            df['Date'] = pd.to_datetime(df['Date'])
            one_year_ago = datetime.now() - timedelta(days=365)
            df = df[df['Date'] >= one_year_ago]
            return df[['Date', 'Close']].rename(columns={'Close': symbol})
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Lỗi Stooq ({symbol}): {e}")
        return pd.DataFrame()

def crawl_all_silver_macro():
    print("⏳ Đang thu thập dữ liệu Bạc và Vĩ mô (Hybrid Source)...")
    
    # 1. Lấy dữ liệu BẠC (XAGUSD) từ Stooq - cái này đã hoạt động
    print("📡 Đang tải XAGUSD từ Stooq...")
    silver_df = get_stooq_data('XAGUSD')
    if silver_df.empty:
        print("❌ Không lấy được giá Bạc từ Stooq.")
        return
    silver_df = silver_df.rename(columns={'XAGUSD': 'silver_usd'})

    # 2. TẬN DỤNG dữ liệu Vĩ mô đã có trong folder gold (Để tránh bị Stooq chặn)
    print("📂 Đang nạp dữ liệu Vĩ mô từ hệ thống Gold...")
    try:
        # Load Macro (DXY, Yield, Gold)
        macro_path = "gold/gold/data/macro_history.csv"
        macro_local = pd.read_csv(macro_path)
        macro_local['Date'] = pd.to_datetime(macro_local['datetime'])
        # Rename columns to match silver project
        macro_local = macro_local.rename(columns={
            'dxy': 'dxy',
            'us_10y_yield': 'us_10y_yield',
            'xau_usd': 'gold_usd'
        })
        
        # Load USDVND
        usdvnd_path = "gold/gold/data/usd_vnd_history.csv"
        usdvnd_local = pd.read_csv(usdvnd_path)
        # Check column name in usd_vnd_history.csv
        date_col = 'date' if 'date' in usdvnd_local.columns else usdvnd_local.columns[0]
        val_col = 'usd_vnd' if 'usd_vnd' in usdvnd_local.columns else usdvnd_local.columns[1]
        usdvnd_local['Date'] = pd.to_datetime(usdvnd_local[date_col])
        usdvnd_local = usdvnd_local[['Date', val_col]].rename(columns={val_col: 'usd_vnd'})
        
        # Merge local data
        local_data = pd.merge(macro_local[['Date', 'dxy', 'us_10y_yield', 'gold_usd']], 
                              usdvnd_local, on='Date', how='inner')
        
        # Merge with Silver
        final_df = pd.merge(silver_df, local_data, on='Date', how='inner')
        
    except Exception as e:
        print(f"⚠️ Không thể nạp dữ liệu local: {e}. Đang thử lấy Gold dự phòng từ Stooq...")
        gold_df = get_stooq_data('XAUUSD')
        if not gold_df.empty:
            final_df = pd.merge(silver_df, gold_df.rename(columns={'XAUUSD': 'gold_usd'}), on='Date', how='outer')
        else:
            final_df = silver_df

    # 3. Làm sạch & Lưu
    final_df = final_df.sort_values('Date').ffill().bfill()
    os.makedirs("silver/data", exist_ok=True)
    final_df.to_csv("silver/data/raw_macro.csv", index=False)
    
    print(f"✅ Đã lưu dữ liệu Master Bạc tại: silver/data/raw_macro.csv")
    print(f"📅 Khoảng thời gian: {final_df['Date'].min().date()} -> {final_df['Date'].max().date()}")
    print(f"📊 Các cột đã lấy được: {list(final_df.columns)}")

if __name__ == "__main__":
    crawl_all_silver_macro()
