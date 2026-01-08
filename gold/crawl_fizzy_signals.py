import urllib.request
import json
import ssl
import csv
import os
import time
from datetime import datetime, timedelta

def crawl_full_daily_logic(days_back=365):
    # 1. Cấu hình
    pairs = ["usa_russia", "russia_ukraine", "usa_china", "china_taiwan", "usa_iran", "usa_venezuela"]
    output_file = "gold/gold/data/pizzint_signals_full.csv"
    
    # Headers chuẩn để giả lập trình duyệt
    headers = {
        'accept': '*/*',
        'referer': 'https://www.pizzint.watch/gdelt',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
    }

    # Bỏ qua lỗi chứng chỉ SSL
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # Tạo file và viết header nếu file chưa tồn tại
    if not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'pair', 'instability_index', 'sentiment', 'conflict_count'])

    # Điểm bắt đầu là ngày 03/01/2026 (Ngày đã chắc chắn có dữ liệu chốt)
    start_point = datetime(2026, 1, 3)

    print(f"🚀 Bắt đầu chiến dịch Crawl FULL: {days_back} ngày lùi từ {start_point.strftime('%Y-%m-%d')}")

    # 2. Vòng lặp chính: Duyệt từng ngày
    for i in range(days_back):
        current_date = start_point - timedelta(days=i)
        
        # API cần dateStart (hôm trước) và dateEnd (ngày đang xét) để trả về đúng 1 bản ghi
        d_start = (current_date - timedelta(days=1)).strftime("%Y%m%d")
        d_end = current_date.strftime("%Y%m%d")
        d_csv = current_date.strftime("%Y-%m-%d")
        
        print(f"\n📅 Ngày: {d_csv} (Request: {d_start} -> {d_end})")
        
        # Duyệt qua 6 cặp quốc gia trong ngày đó
        for pair in pairs:
            url = f"https://www.pizzint.watch/api/gdelt?pair={pair}&method=gpr&dateStart={d_start}&dateEnd={d_end}"
            
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15, context=ctx) as response:
                    res_body = response.read().decode('utf-8')
                    data = json.loads(res_body)
                    
                    if data and isinstance(data, list):
                        # Lấy bản ghi đầu tiên trả về
                        record = data[0]
                        
                        # Lưu ngay vào CSV
                        with open(output_file, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                d_csv,
                                pair,
                                record.get('v'),
                                record.get('sentiment'),
                                record.get('conflictCount')
                            ])
                        print(f"   ✅ {pair:<15}: v={record.get('v'):.4f} | Sent={record.get('sentiment'):.2f}")
                    else:
                        print(f"   ⚠️ {pair:<15}: Không có dữ liệu.")
                
                # Nghỉ ngắn giữa các cặp để tránh bị block
                time.sleep(0.3)
                
            except Exception as e:
                print(f"   ❌ {pair:<15}: Lỗi API ({e})")
                # Nếu bị lỗi 429 (Too Many Requests) thì nghỉ lâu hơn
                if "429" in str(e):
                    print("🛑 Bị giới hạn tốc độ. Nghỉ 10s...")
                    time.sleep(10)
        
        # Nghỉ 1 giây sau khi xong 1 ngày
        time.sleep(1)

    print(f"\n✨ HOÀN THÀNH! Dữ liệu đã được lưu đầy đủ tại: {output_file}")

if __name__ == "__main__":
    # Bạn có thể điều chỉnh số ngày muốn lấy ở đây (ví dụ 365 ngày cho 1 năm)
    crawl_full_daily_logic(days_back=365)