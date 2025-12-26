"""
Lấy danh sách mã chứng khoán VN30 từ API
"""
import requests
import json
import os


def get_vn30_symbols(host: str, api_key: str = None, headers: dict = None) -> list:
    """
    Lấy danh sách stockSymbol từ API VN30
    
    Args:
        host: API host
        api_key: API key (sẽ được thêm vào header x-api-key)
        headers: Custom headers (nếu có)
    
    Returns:
        List các stockSymbol
    """
    url = f"{host}/price-api/vnstock/vn30"
    
    # Headers mặc định
    request_headers = {
        'accept': 'application/json',  # Header theo API yêu cầu
    }
    
    # Thêm x-api-key header (theo format API yêu cầu)
    if api_key:
        request_headers['x-api-key'] = api_key
    
    # Thêm custom headers nếu có
    if headers:
        request_headers.update(headers)
    
    print(f"URL: {url}")
    print(f"Headers: {list(request_headers.keys())}")
    
    try:
        # QUAN TRỌNG: Truyền headers vào request
        response = requests.get(url, headers=request_headers, timeout=30)
        
        # Kiểm tra status code
        if response.status_code == 401:
            print("❌ 401 Unauthorized - API cần x-api-key header")
            print("   Vui lòng cung cấp API key")
            print(f"   Response: {response.text[:200]}")
            return []
        
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('code') != 'SUCCESS':
            print(f"❌ API error: {data.get('message')}")
            return []
        
        # Extract stockSymbol từ data array
        symbols = []
        for item in data.get('data', []):
            symbol = item.get('stockSymbol')
            if symbol:
                symbols.append(symbol)
        
        print(f"✓ Thành công: Lấy được {len(symbols)} mã")
        return symbols
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Status: {e.response.status_code}")
            print(f"   Response: {e.response.text[:200]}")
        return []
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    HOST = "https://price.beaverx.ai"
    
    # Lấy API key từ environment variable
    API_KEY = os.getenv('API_KEY') or os.getenv('BEAVERX_API_KEY')
    
    # Hoặc hardcode (chỉ để test, không khuyến nghị)
    # API_KEY = "your-api-key-here"
    
    if API_KEY:
        print(f"Đang sử dụng API key...")
        symbols = get_vn30_symbols(HOST, api_key=API_KEY)
    else:
        print("⚠️ Không có API key, thử gọi không authentication...")
        symbols = get_vn30_symbols(HOST)
    
    if symbols:
        print(f"\nDanh sách {len(symbols)} mã VN30:")
        print(symbols)
        
        # Lưu vào file
        with open("vn30_symbols.txt", "w", encoding="utf-8") as f:
            for symbol in symbols:
                f.write(f"{symbol}\n")
        
        print(f"\n✓ Đã lưu vào vn30_symbols.txt")
        
    else:
        print("\n❌ Không lấy được danh sách mã")
        print("\nHướng dẫn:")
        print("1. Set environment variable: export API_KEY='your-key'")
        print("2. Hoặc sửa code để thêm API key trực tiếp")