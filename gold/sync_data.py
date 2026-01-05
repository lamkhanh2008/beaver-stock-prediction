import os
import csv
from datetime import datetime

def sync_master_with_sjc():
    print("🧹 Đang đồng bộ hóa file Master với dữ liệu thực tế từ sjc.csv...")
    
    SJC_FILE = os.path.join("gold", "gold", "data", "sjc.csv")
    MASTER_FILE = os.path.join("gold", "gold", "data", "gold_master_history.csv")
    
    if not os.path.exists(SJC_FILE):
        print("❌ Không tìm thấy file sjc.csv")
        return

    # 1. Đọc dữ liệu từ sjc.csv (Lấy giá mới nhất)
    sjc_data = {}
    last_date = ""
    last_price = 0.0
    with open(SJC_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            if not row or len(row) < 4: continue
            try:
                date_str = row[0].split(' ')[0]
                sell_price = float(row[3])
                sjc_data[date_str] = sell_price
                last_date = date_str
                last_price = sell_price
            except: continue

    # 2. Đọc và sửa file Master
    if not os.path.exists(MASTER_FILE):
        print("❌ Không tìm thấy file Master.")
        return

    with open(MASTER_FILE, 'r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))

    updated_rows = []
    REAL_WORLD_GOLD = 3650.0 
    FX_RATE = 26292.0

    for r in reader:
        date = r['date']
        
        # Nếu ngày này có trong SJC.csv -> Cập nhật đúng giá
        if date in sjc_data:
            r['sjc_price'] = str(sjc_data[date])
            updated_rows.append(r)
        # Nếu ngày này là ngày "tương lai" (do script update tạo ra) 
        # -> Ta sẽ xóa nó đi để Master file luôn khớp với dữ liệu thực tế của bạn
        else:
            print(f"⚠️ Loại bỏ ngày {date} vì không có trong sjc.csv")
            continue

    # 3. Ghi đè lại
    fieldnames = ['date', 'fed', 'dxy', 'xau_usd', 'xau_vnd_tael', 'sjc_price']
    with open(MASTER_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"✅ Đã đồng bộ xong. Giá SJC cuối cùng: {last_price} (Ngày {last_date})")

if __name__ == "__main__":
    sync_master_with_sjc()
