import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime

def crawl_giavang_asia_silver():
    print("⏳ Đang thu thập lịch sử giá bạc Việt Nam từ giavang.asia...")
    
    url = "https://giavang.asia/gia-bac/"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            print(f"❌ Không thể truy cập (Status: {response.status_code})")
            return pd.DataFrame()
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Trang này thường có các bảng giá hiện tại và biểu đồ. 
        # Nếu không có bảng lịch sử dài hạn, ta sẽ lấy giá hiện tại và kết hợp với dữ liệu thế giới.
        # Tuy nhiên, ta thử tìm các bảng có class 'table'
        tables = soup.find_all('table')
        
        rows = []
        for table in tables:
            for tr in table.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) >= 3:
                    text = tr.text.lower()
                    if 'bạc' in text or 'silver' in text:
                        # Thử parse giá
                        name = tds[0].text.strip()
                        buy = tds[1].text.strip().replace('.', '').replace(',', '')
                        sell = tds[2].text.strip().replace('.', '').replace(',', '')
                        
                        try:
                            rows.append({
                                'name': name,
                                'buy': float(buy) if buy.isdigit() else 0,
                                'sell': float(sell) if sell.isdigit() else 0,
                                'date': datetime.now().date()
                            })
                        except: continue
        
        if rows:
            print(f"✅ Đã lấy được giá bạc hôm nay: {rows[0]['sell']} VNĐ")
            return pd.DataFrame(rows)
        return pd.DataFrame()
            
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = crawl_giavang_asia_silver()
    if not df.empty:
        print(df)


