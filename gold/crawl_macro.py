import os
import requests
import csv
import sys
from datetime import datetime

# Import API keys from config
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config.settings import FRED_API_KEY
except ImportError:
    print("⚠️ Cảnh báo: Không tìm thấy config/settings.py. Vui lòng tạo file từ settings_example.py")
    FRED_API_KEY = os.getenv("FRED_API_KEY", "")


def fetch_fred_data(series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&sort_order=desc&limit=1"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if 'observations' in data and len(data['observations']) > 0:
            val = data['observations'][0]['value']
            return float(val) if val != "." else None
    except: return None
    return None

def update_macro_history():
    print("🚀 Đang đồng bộ dữ liệu Vĩ mô & Vàng Thế giới (XAU/USD)...")
    
    # Lấy dữ liệu mới nhất
    dxy = fetch_fred_data("DTWEXBGS")
    fed = fetch_fred_data("DFF")
    xau_usd = fetch_fred_data("GOLDAMGBD228NLBM") # Giá vàng thế giới chuẩn
    usd_vnd = fetch_fred_data("DEXVNM")

    today = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join("gold", "gold", "data", "macro_history.csv")
    
    # Đọc dữ liệu cũ để tránh trùng
    rows = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            rows = list(csv.reader(f))
    
    # Cập nhật hoặc thêm mới (Cấu trúc: date, fed, yield, dxy, xau_usd)
    # Ta sẽ ghi đè dòng cuối nếu là ngày hôm nay, hoặc append nếu là ngày mới
    new_row = [today, fed or 3.64, 4.12, dxy or 120.0, xau_usd or 4375.0]
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["datetime", "fed_rate", "us_10y_yield", "dxy", "xau_usd"])
        for r in rows[1:]:
            if r[0] != today: writer.writerow(r)
        writer.writerow(new_row)
    print(f"✅ Đã cập nhật XAU/USD: {xau_usd} USD/ounce")

if __name__ == "__main__":
    update_macro_history()
