import pandas as pd
import os
from datetime import datetime, timedelta

def rebuild_master_clean_1y():
    BASE_PATH = "gold/gold/data"
    print("🧹 Bắt đầu chuẩn hóa dữ liệu: 1 ngày/dòng, liên tục 365 ngày...")
    
    # 1. Load các file thành phần
    macro_df = pd.read_csv(os.path.join(BASE_PATH, "macro_history.csv"))
    world_gold_df = pd.read_csv(os.path.join(BASE_PATH, "world_gold_real_vnd.csv"))
    sjc_df = pd.read_csv(os.path.join(BASE_PATH, "sjc.csv"))
    
    # Chuẩn hóa format ngày
    macro_df['date'] = pd.to_datetime(macro_df['datetime']).dt.date
    world_gold_df['date'] = pd.to_datetime(world_gold_df['date']).dt.date
    sjc_df['date'] = pd.to_datetime(sjc_df['datetime']).dt.date
    
    # 2. Tạo khung 365 ngày liên tục (Kết thúc là ngày mới nhất có trong SJC)
    end_date = sjc_df['date'].max()
    start_date = end_date - timedelta(days=364)
    all_dates = pd.date_range(start=start_date, end=end_date).date
    master_df = pd.DataFrame({'date': all_dates})
    
    # 3. Xử lý từng nguồn dữ liệu trước khi gộp
    # Macro
    macro_clean = macro_df.groupby('date').last().reset_index()
    macro_clean = macro_clean.rename(columns={'fed_rate': 'fed'})
    
    # World Gold
    world_clean = world_gold_df.groupby('date').last().reset_index()
    world_clean = world_clean.rename(columns={'price_vnd_tael': 'xau_vnd_tael'})
    
    # SJC (Lấy giá Sell)
    sjc_clean = sjc_df.groupby('date').last().reset_index()
    sjc_clean = sjc_clean.rename(columns={'sell': 'sjc_price'})
    
    # 4. Gộp tất cả vào khung Master (Left Join để không mất ngày)
    master_df = pd.merge(master_df, macro_clean[['date', 'fed', 'us_10y_yield', 'dxy', 'xau_usd']], on='date', how='left')
    master_df = pd.merge(master_df, world_clean[['date', 'xau_vnd_tael']], on='date', how='left')
    master_df = pd.merge(master_df, sjc_clean[['date', 'sjc_price']], on='date', how='left')
    
    # 5. LẤP ĐẦY KHOẢNG TRỐNG (Forward Fill)
    # Nếu cuối tuần/ngày lễ không có giá, lấy giá ngày trước đó. 
    # Điều này cực kỳ quan trọng để AI không bị lỗi NaN.
    master_df = master_df.sort_values('date').ffill().bfill()
    
    # 6. Kiểm tra và Lưu
    output_path = os.path.join(BASE_PATH, "gold_master_history.csv")
    master_df.to_csv(output_path, index=False)
    
    print(f"✅ Đã tạo xong file Master sạch!")
    print(f"📅 Khoảng thời gian: {master_df['date'].min()} đến {master_df['date'].max()}")
    print(f"📊 Tổng số ngày: {len(master_df)} (Đúng 1 năm)")
    print(f"🔎 Cột dữ liệu: {list(master_df.columns)}")
    print(f"📈 Dòng cuối cùng:\n{master_df.tail(1)}")

if __name__ == "__main__":
    rebuild_master_clean_1y()
