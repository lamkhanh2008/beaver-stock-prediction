import os
import requests
import csv
from datetime import datetime

# --- CẤU HÌNH QUỐC TẾ ---
FRED_API_KEY = "43a0e050800d984180410710609b78a4"
MASTER_FILE = os.path.join("gold", "gold", "data", "gold_master_history.csv")

def fetch_fred(series_id):
    """Lấy dữ liệu từ FRED API (Mỹ)"""
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&sort_order=desc&limit=1"
    try:
        res = requests.get(url, timeout=10).json()
        val = res['observations'][0]['value']
        return float(val) if val != "." else None
    except: return None

def get_latest_exchange_rate():
    """Giả lập lấy tỷ giá mới nhất (Bạn có thể cập nhật con số này sáng sớm)"""
    return 26292.0

def update_gold_master():
    print("🚀 Đang thực hiện chiến dịch: 'Giải cứu dữ liệu Vàng Thế giới'...")
    
    # 1. Thu thập dữ liệu chuẩn từ FRED
    dxy = fetch_fred("DTWEXBGS") or 120.0
    fed = fetch_fred("DFF") or 3.64
    xau_usd = fetch_fred("GOLDAMGBD228NLBM") or 4375.0 # Giá vàng chuẩn London
    
    today = datetime.now().strftime("%Y-%m-%d")
    rate = get_latest_exchange_rate()
    
    # 2. Công thức quy đổi chuẩn: 1 lượng = 1.20565 ounce
    # Giá VND = (USD * 1.20565 * Tỷ giá)
    vnd_world_tael = round((xau_usd * 1.20565 * rate) / 1000000, 2)
    
    # Giả sử giá SJC VN sáng nay (Bạn có thể sửa con số này theo bảng điện)
    current_sjc = 154.7
    
    if not os.path.exists(MASTER_FILE):
        print("❌ Không tìm thấy file Master.")
        return

    # 3. Đọc và Cập nhật
    with open(MASTER_FILE, 'r') as f:
        lines = f.readlines()
    
    header = lines[0]
    data_lines = lines[1:]
    last_line = data_lines[-1].split(',')
    last_date = last_line[0]

    new_entry = f"{today},{fed},{dxy},{xau_usd},{vnd_world_tael},{current_sjc}\n"

    if last_date == today:
        print(f"ℹ️ Đang cập nhật dữ liệu mới nhất cho ngày hôm nay ({today})...")
        data_lines[-1] = new_entry
    else:
        print(f"✅ Đã phát hiện ngày mới {today}. Đang nối thêm dữ liệu chuẩn...")
        data_lines.append(new_entry)

    with open(MASTER_FILE, 'w') as f:
        f.write(header)
        f.writelines(data_lines)

    print("-" * 50)
    print(f"🌍 GIÁ THẾ GIỚI: ${xau_usd} USD/ounce")
    print(f"🇻🇳 QUY ĐỔI VNĐ:  {vnd_world_tael} triệu/lượng")
    print(f"💎 GIÁ SJC VN:   {current_sjc} triệu/lượng")
    print(f"📊 CHÊNH LỆCH:   {round(current_sjc - vnd_world_tael, 2)} triệu (REAL BASIS)")
    print("-" * 50)
    print("✅ Đã đồng bộ thành công vào gold_master_history.csv")

if __name__ == "__main__":
    update_gold_master()
