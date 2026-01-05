import requests
import re
import os

def get_news_shock_score():
    """Quét tin tức nóng từ các trang tài chính để tìm cú sốc địa chính trị"""
    print("📰 Đang quét tin tức chấn động thế giới...")
    
    # Danh sách các nguồn tin (Bạn có thể thêm nhiều nguồn)
    sources = [
        "https://tuoitre.vn/the-gioi.htm",
        "https://vnexpress.net/the-gioi",
        "https://cafef.vn/the-gioi.chn"
    ]
    
    # Từ khóa chấn động
    shock_keywords = [
        "bắt giữ", "bắt", "đảo chính", "chiến tranh", "tấn công", 
        "xung đột", "căng thẳng", "bom", "tên lửa", "lật đổ", 
        "khẩn cấp", "venezuela", "nga", "ukraine", "trung đông"
    ]
    
    total_score = 0
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for url in sources:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                # Tìm tất cả tiêu đề (thường nằm trong thẻ <a> hoặc <h3>)
                content = response.text.lower()
                for word in shock_keywords:
                    count = len(re.findall(word, content))
                    if count > 0:
                        total_score += count * 0.5 # Mỗi lần xuất hiện cộng điểm rủi ro
        except:
            continue
            
    # Chuẩn hóa điểm số: Nếu điểm > 5 coi như có biến động lớn
    print(f"📊 Điểm rủi ro tin tức hiện tại: {total_score}")
    return True if total_score > 5 else False

if __name__ == "__main__":
    is_shock = get_news_shock_score()
    if is_shock:
        print("🔥 CẢNH BÁO: Phát hiện tin tức chấn động! Kích hoạt chế độ Safe Haven.")
    else:
        print("❄️ Tin tức ổn định.")

