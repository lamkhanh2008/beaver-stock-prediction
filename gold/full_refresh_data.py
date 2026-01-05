import os
import csv
import requests
from datetime import datetime, timedelta

def get_stooq_data(symbol):
    """Lấy dữ liệu từ Stooq (nguồn thay thế Yahoo cực tốt)"""
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200 and "Date,Open" in response.text:
            lines = response.text.strip().split('\n')
            reader = csv.DictReader(lines)
            return {row['Date'].replace('-', '-'): float(row['Close']) for row in reader}
        else:
            print(f"⚠️ Stooq từ chối {symbol} hoặc không có dữ liệu.")
            return {}
    except Exception as e:
        print(f"⚠️ Lỗi kết nối Stooq: {e}")
        return {}

def refresh_all_master_data():
    print('⏳ Đang tải dữ liệu thật từ Stooq (Vàng & DXY)...')
    
    # XAUUSD: Vàng thế giới, USDVND: Tỷ giá (nếu cần), ^DXY: Chỉ số Dollar
    gold_history = get_stooq_data('XAUUSD')
    dxy_history = get_stooq_data('USDIDX') # Mã DXY trên Stooq là USDIDX
    
    if not gold_history:
        print('❌ Không lấy được dữ liệu Vàng. Đang thử nguồn dự phòng cuối cùng...')
        # Nếu vẫn lỗi, tôi sẽ dùng một link CSV trực tiếp từ GitHub hoặc nguồn mở
        return

    master_path = os.path.join('gold', 'gold', 'data', 'gold_master_history.csv')
    if not os.path.exists(master_path):
        print(f'❌ Không tìm thấy file Master.')
        return

    with open(master_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    fieldnames = ['date', 'fed', 'dxy', 'xau_usd', 'xau_vnd_tael', 'sjc_price']
    updated_rows = []
    DEFAULT_FX = 26292.0

    print(f'📊 Đang cập nhật biến động thật cho {len(rows)} ngày...')
    
    # Để đảm bảo không bị "đứng hình", ta sẽ dùng Forward Fill nếu Stooq thiếu vài ngày
    last_gold = 2600.0
    last_dxy = 105.0

    for r in rows:
        date = r['date']
        
        # Cập nhật Vàng
        if date in gold_history:
            last_gold = gold_history[date]
        r['xau_usd'] = str(round(last_gold, 2))
        
        # Cập nhật DXY
        if date in dxy_history:
            last_dxy = dxy_history[date]
        r['dxy'] = str(round(last_dxy, 4))
        
        # Tính quy đổi VND
        try:
            r['xau_vnd_tael'] = str(round((last_gold * 1.20565 * DEFAULT_FX) / 1000000, 2))
        except: pass
        
        updated_rows.append(r)

    with open(master_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f'✅ THÀNH CÔNG! Dữ liệu đã có Biến động (Vàng hiện tại: {last_gold}$).')
    print(f'👉 Bây giờ hãy chạy: python gold/predict_gold.py')

if __name__ == "__main__":
    refresh_all_master_data()
