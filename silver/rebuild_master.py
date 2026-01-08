import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def rebuild_silver_master():
    BASE_PATH = "silver/data"
    print("🧹 Đang xây dựng Master Data cho Bạc Việt Nam...")
    
    # 1. Load raw macro
    if not os.path.exists(os.path.join(BASE_PATH, "raw_macro.csv")):
        print("❌ Không thấy file raw_macro.csv. Hãy chạy crawl_data.py trước.")
        return
        
    df = pd.read_csv(os.path.join(BASE_PATH, "raw_macro.csv"))
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Tạo khung thời gian liên tục 365 ngày (để tránh mất ngày lễ/cuối tuần)
    end_date = df['Date'].max()
    start_date = end_date - timedelta(days=364)
    all_dates = pd.date_range(start=start_date, end=end_date)
    master_df = pd.DataFrame({'date': all_dates})
    
    # Merge dữ liệu vào khung
    df = df.rename(columns={'Date': 'date'})
    master_df = pd.merge(master_df, df, on='date', how='left')
    
    # 3. Lấp đầy dữ liệu bị khuyết (Forward Fill)
    master_df = master_df.ffill().bfill()
    
    # 4. TÍNH GIÁ BẠC VIỆT NAM (Mua vào & Bán ra - Phú Quý)
    # Cột 'silver_usd' thực chất là giá Bạc VN (nghìn VNĐ/kg)
    # Quy đổi: 1 kg = 26.6667 lượng
    
    # Giá Bán ra (triệu/lượng) - Lấy trực tiếp từ dữ liệu gốc
    master_df['silver_vn_sell'] = (master_df['silver_usd'] / 26.6667)
    
    # Giá Mua vào (triệu/lượng) - Thường thấp hơn giá bán khoảng 3% (theo biểu đồ Phú Quý)
    # Ví dụ: 82.5 triệu bán ra -> 80.0 triệu mua vào (Spread ~2.5 triệu/kg)
    master_df['silver_vn_buy'] = master_df['silver_vn_sell'] * 0.97
    
    # Giữ cột silver_vn_price làm trung bình để các script cũ không bị lỗi
    master_df['silver_vn_price'] = (master_df['silver_vn_buy'] + master_df['silver_vn_sell']) / 2
    
    # 5. Thêm Gold-Silver Ratio (GSR)
    master_df['gsr'] = master_df['gold_usd'] / (master_df['silver_usd'] + 1e-8)
    
    # 6. Lưu file Master
    output_path = os.path.join(BASE_PATH, "silver_master_history.csv")
    master_df.to_csv(output_path, index=False)
    
    print(f"✅ Đã tạo xong Silver Master Data!")
    print(f"📊 Số lượng mẫu: {len(master_df)} ngày.")
    print(f"💰 Giá Bạc VN hiện tại (ước tính): {master_df['silver_vn_price'].iloc[-1]:.2f} triệu/lượng")
    print(f"📈 Cột dữ liệu: {list(master_df.columns)}")

if __name__ == "__main__":
    rebuild_silver_master()

