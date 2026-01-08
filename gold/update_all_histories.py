import csv
import os
from datetime import datetime, timedelta

def update_auxiliary_files():
    print("🚀 Đang đồng bộ hóa tất cả các file dữ liệu vĩ mô và tỷ giá đến ngày 06/01/2026...")
    
    BASE_PATH = "gold/gold/data"
    files_to_update = {
        "usd_vnd_history.csv": ["date", "usd_vnd"],
        "macro_history.csv": ["datetime", "fed_rate", "us_10y_yield", "dxy", "xau_usd"],
        "world_gold_real_vnd.csv": ["date", "price_usd", "fx_rate", "price_vnd_tael"]
    }
    
    target_dates = ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07"]
    
    for filename, headers in files_to_update.items():
        file_path = os.path.join(BASE_PATH, filename)
        if not os.path.exists(file_path):
            print(f"⚠️ Không tìm thấy {filename}, bỏ qua.")
            continue
            
        # Đọc dữ liệu cũ
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = list(csv.reader(f))
            header = lines[0]
            data = lines[1:]
        
        last_row = data[-1]
        last_date = last_row[0].split(' ')[0]
        
        # Nếu chưa đến ngày 04/01, ta tiến hành append
        new_rows = []
        for d in target_dates:
            # Kiểm tra xem ngày đã tồn tại chưa
            if any(d in row[0] for row in data):
                continue
                
            # Tạo dòng mới dựa trên giá trị của dòng cuối cùng (Forward Fill)
            new_row = [d] + last_row[1:]
            new_rows.append(new_row)
            
        if new_rows:
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(new_rows)
            print(f"✅ Đã cập nhật {len(new_rows)} ngày mới cho {filename}.")
        else:
            print(f"ℹ️ {filename} đã đầy đủ dữ liệu đến {last_date}.")

    print("✨ Tất cả các file thành phần đã được đồng bộ!")

if __name__ == "__main__":
    update_auxiliary_files()

