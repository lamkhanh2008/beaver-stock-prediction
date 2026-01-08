import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime

def crawl_vn_silver_history():
    print("⏳ Đang thu thập lịch sử giá bạc thực tế tại Việt Nam từ exchange-rates.org...")
    
    url = "https://www.exchange-rates.org/vn/kim-loai-quy/bac/viet-nam/lich-su-gia"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            print(f"❌ Không thể truy cập trang web (Status: {response.status_code})")
            return pd.DataFrame()
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Tìm bảng lịch sử giá
        table = soup.find('table')
        if not table:
            print("❌ Không tìm thấy bảng dữ liệu trên trang web.")
            return pd.DataFrame()
            
        rows = []
        # Duyệt qua các hàng của bảng (bỏ qua header)
        for tr in table.find_all('tr')[1:]:
            tds = tr.find_all('td')
            if len(tds) >= 2:
                # Cấu trúc: [Ngày, Giá VND/Ounce]
                date_str = tds[0].text.strip()
                price_str = tds[1].text.strip().replace('₫', '').replace(',', '').replace('.', '')
                
                try:
                    # Chuyển đổi ngày (format trên trang này thường là DD/MM/YYYY)
                    dt = pd.to_datetime(date_str, dayfirst=True)
                    # Chuyển đổi giá sang triệu VND/lượng
                    # 1 Ounce = 0.8294 lượng (tương đối). 
                    # Để đơn giản và chuẩn xác cho AI, ta lấy giá Ounce VND rồi quy đổi
                    price_vnd_ounce = float(price_str)
                    price_million_tael = (price_vnd_ounce * 1.20565) / 1000000
                    
                    rows.append({
                        'Date': dt,
                        'silver_vn_real': round(price_million_tael, 3)
                    })
                except Exception as e:
                    continue
                    
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('Date')
            print(f"✅ Đã lấy được {len(df)} ngày dữ liệu giá bạc thực tế.")
            return df
        else:
            print("❌ Dữ liệu trống.")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"⚠️ Lỗi khi crawl giá bạc VN: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    vn_silver = crawl_vn_silver_history()
    if not vn_silver.empty:
        os.makedirs("silver/data", exist_ok=True)
        vn_silver.to_csv("silver/data/vn_silver_history.csv", index=False)
        print("✅ Đã lưu: silver/data/vn_silver_history.csv")


