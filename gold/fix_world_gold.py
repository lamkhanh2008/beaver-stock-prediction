import os
import requests
import csv

# API lấy lịch sử 1 năm
FRED_API_KEY = "43a0e050800d984180410710609b78a4"
MASTER_FILE = os.path.join("gold", "gold", "data", "gold_master_history.csv")

def fix_all_history():
    print("🧹 Đang dọn dẹp và sửa lại toàn bộ dữ liệu vàng thế giới trong lịch sử...")
    
    # 1. Lấy lịch sử XAU/USD từ FRED (Series: GOLDAMGBD228NLBM)
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=GOLDAMGBD228NLBM&api_key={FRED_API_KEY}&file_type=json"
    try:
        res = requests.get(url, timeout=20).json()
        world_history = {obs['date']: float(obs['value']) for obs in res['observations'] if obs['value'] != "."}
    except:
        print("❌ Lỗi mạng. Hãy đảm bảo bạn có kết nối internet và requests đã cài đặt.")
        return

    if not os.path.exists(MASTER_FILE): return

    # 2. Đọc file Master hiện tại
    with open(MASTER_FILE, 'r') as f:
        reader = list(csv.DictReader(f))

    # 3. Cập nhật lại cột xau_usd và xau_vnd_tael chuẩn
    updated_rows = []
    for row in reader:
        date = row['date']
        if date in world_history:
            real_usd = world_history[date]
            # Quy đổi dựa trên tỷ giá ngày hôm đó trong file
            rate = float(row.get('fx_rate') or 26292.0)
            vnd_tael = round((real_usd * 1.20565 * rate) / 1000000, 2)
            
            row['xau_usd'] = real_usd
            row['xau_vnd_tael'] = vnd_tael
        updated_rows.append(row)

    # 4. Ghi đè lại
    with open(MASTER_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=reader[0].keys())
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"✅ Đã dọn dẹp xong {len(updated_rows)} ngày. Dữ liệu CafeF cũ đã bị xóa bỏ hoàn toàn!")

if __name__ == "__main__":
    fix_all_history()

