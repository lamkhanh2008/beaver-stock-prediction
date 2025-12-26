"""
Crawl dữ liệu VN Stock từ API và lưu thành CSV
"""
import datetime
import requests
import pandas as pd
import json
import time
import os
from typing import Optional, Dict


class VNStockCrawler:
    """Class để crawl dữ liệu cổ phiếu Việt Nam"""
    
    def __init__(self, host: str, output_dir: str = "data", api_key: str = None):
        """
        Args:
            host: API host (ví dụ: "https://api.example.com")
            output_dir: Thư mục lưu dữ liệu
            api_key: API key (nếu None sẽ lấy từ environment variable)
        """
        self.host = host.rstrip('/')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Lấy API key từ parameter hoặc environment variable
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('API_KEY') or os.getenv('BEAVERX_API_KEY')
        
    def get_timestamp_range(self, years_back: int = 3) -> tuple:
        """Tính timestamp từ hiện tại đến N năm trước"""
        now = datetime.datetime.now()
        start_date = now - datetime.timedelta(days=years_back * 365)
        
        # Đặt về 00:00:00
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        now = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        timestamp_from = int(start_date.timestamp())
        timestamp_to = int(now.timestamp())
        
        return timestamp_from, timestamp_to, start_date, now
    
    def fetch_stock_data(self, symbol: str, resolution: str = "1D", 
                        timestamp_from: int = None, timestamp_to: int = None,
                        years_back: int = 3) -> Optional[Dict]:
        """
        Fetch dữ liệu từ API
        
        Args:
            symbol: Mã cổ phiếu (ví dụ: 'ACB')
            resolution: Độ phân giải ('1D' = daily)
            timestamp_from: Timestamp bắt đầu
            timestamp_to: Timestamp kết thúc
            years_back: Số năm lùi lại (nếu không có timestamp)
        
        Returns:
            Response data hoặc None nếu lỗi
        """
        # Tính timestamp nếu chưa có
        if timestamp_from is None or timestamp_to is None:
            ts_from, ts_to, _, _ = self.get_timestamp_range(years_back)
            timestamp_from = timestamp_from or ts_from
            timestamp_to = timestamp_to or ts_to
        
        # Headers mặc định
        request_headers = {
            'accept': 'application/json',
        }
        
        # Thêm x-api-key header (theo format API yêu cầu)
        if self.api_key:
            request_headers['x-api-key'] = self.api_key
        
        url = f"{self.host}/price-api/vnstock-data/charts"
        params = {
            "resolution": resolution,
            "symbol": symbol,
            "from": timestamp_from,
            "to": timestamp_to
        }
        
        print(f"Đang crawl {symbol}...")
        print(f"  From: {datetime.datetime.fromtimestamp(timestamp_from).strftime('%Y-%m-%d')}")
        print(f"  To: {datetime.datetime.fromtimestamp(timestamp_to).strftime('%Y-%m-%d')}")
        
        try:
            response = requests.get(url, headers=request_headers, params=params, timeout=30)
            
            # Kiểm tra status code
            if response.status_code == 401:
                print(f"  ❌ 401 Unauthorized - API cần x-api-key header")
                print(f"  Vui lòng set: export API_KEY='your-key'")
                print(f"  Response: {response.text[:200]}")
                return None
            
            response.raise_for_status()
            
            data = response.json()
            print(f"  ✓ Thành công: {len(str(data))} bytes")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Lỗi API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"  Status: {e.response.status_code}")
                print(f"  Response: {e.response.text[:200]}")
            return None
        except json.JSONDecodeError as e:
            print(f"  ❌ Lỗi parse JSON: {e}")
            if 'response' in locals():
                print(f"  Response text: {response.text[:200]}")
            return None
    
    def parse_chart_data(self, data: Dict, symbol: str) -> Optional[pd.DataFrame]:
        """
        Parse dữ liệu từ API response thành DataFrame
        
        Format response: {"data": {"t": [...], "c": [...], "o": [...], "h": [...], "l": [...], "v": [...]}}
        """
        df = None
        
        # Format: Response có 'data' key chứa object với arrays
        if 'data' in data and isinstance(data['data'], dict):
            data_dict = data['data']
            # Kiểm tra xem có arrays không
            if 't' in data_dict and 'c' in data_dict:
                try:
                    timestamps = data_dict.get('t', [])
                    opens = data_dict.get('o', [])
                    highs = data_dict.get('h', [])
                    lows = data_dict.get('l', [])
                    closes = data_dict.get('c', [])
                    volumes = data_dict.get('v', [])
                    
                    # Kiểm tra độ dài arrays
                    lengths = {
                        't': len(timestamps),
                        'o': len(opens),
                        'h': len(highs),
                        'l': len(lows),
                        'c': len(closes),
                        'v': len(volumes) if volumes else 0
                    }
                    
                    # Lấy độ dài tối thiểu (tránh lỗi khi arrays không bằng nhau)
                    min_length = min([l for l in lengths.values() if l > 0])
                    
                    # Cắt tất cả arrays về cùng độ dài
                    timestamps = timestamps[:min_length]
                    opens = opens[:min_length] if opens else [None] * min_length
                    highs = highs[:min_length] if highs else [None] * min_length
                    lows = lows[:min_length] if lows else [None] * min_length
                    closes = closes[:min_length] if closes else [None] * min_length
                    volumes = volumes[:min_length] if volumes else [0] * min_length
                    
                    # Tạo DataFrame
                    df = pd.DataFrame({
                        'Open': opens,
                        'High': highs,
                        'Low': lows,
                        'Close': closes,
                        'Volume': volumes
                    })
                    
                    # Convert timestamp to datetime
                    df.index = pd.to_datetime(timestamps, unit='s')
                    print(f"  ✓ Parse format: data object với arrays")
                    print(f"  Arrays length: {min_length}")
                    
                except Exception as e:
                    print(f"  ⚠️ Lỗi parse format data object: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Format 1: Response trực tiếp có 't', 'o', 'h', 'l', 'c', 'v' arrays
        elif 't' in data and 'c' in data:
            try:
                timestamps = data.get('t', [])
                df = pd.DataFrame({
                    'Open': data.get('o', []),
                    'High': data.get('h', []),
                    'Low': data.get('l', []),
                    'Close': data.get('c', []),
                    'Volume': data.get('v', [0] * len(timestamps))  # Default 0 nếu không có
                })
                # Convert timestamp to datetime
                df.index = pd.to_datetime(timestamps, unit='s')
                print(f"  ✓ Parse format: arrays (t, o, h, l, c, v)")
            except Exception as e:
                print(f"  ⚠️ Lỗi parse format arrays: {e}")
        
        # Format 2: Response là array of objects
        elif isinstance(data, list) and len(data) > 0:
            try:
                records = []
                for item in data:
                    if isinstance(item, dict):
                        record = {
                            'Open': item.get('open') or item.get('o'),
                            'High': item.get('high') or item.get('h'),
                            'Low': item.get('low') or item.get('l'),
                            'Close': item.get('close') or item.get('c'),
                            'Volume': item.get('volume') or item.get('v') or 0
                        }
                        # Timestamp
                        ts = item.get('time') or item.get('t') or item.get('timestamp')
                        if ts:
                            record['timestamp'] = ts
                        records.append(record)
                
                if records:
                    df = pd.DataFrame(records)
                    if 'timestamp' in df.columns:
                        df.index = pd.to_datetime(df['timestamp'], unit='s')
                        df = df.drop(columns=['timestamp'])
                    print(f"  ✓ Parse format: array of objects")
            except Exception as e:
                print(f"  ⚠️ Lỗi parse format array: {e}")
        
        # Format 3: Response có 'result' key
        elif 'result' in data:
            return self.parse_chart_data(data['result'], symbol)
        
        if df is None or df.empty:
            print(f"  ⚠️ Không parse được dữ liệu")
            print(f"  Data type: {type(data)}")
            print(f"  Data keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            if isinstance(data, dict) and 'data' in data:
                print(f"  Data['data'] keys: {list(data['data'].keys()) if isinstance(data['data'], dict) else 'N/A'}")
            print(f"  Sample: {str(data)[:500]}")
            return None
        
        # Clean và validate
        # Xử lý Volume: nếu không có hoặc None thì set = 0
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        
        # Đảm bảo có đủ columns cần thiết
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"  ❌ Thiếu columns bắt buộc: {missing}")
            return None
        
        # Convert về numeric và drop NaN
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows có NaN ở các columns bắt buộc
        df = df.dropna(subset=required_cols)
        
        # Nếu không có Volume, tạo cột Volume = 0
        if 'Volume' not in df.columns:
            df['Volume'] = 0
            print(f"  ⚠️ Không có Volume, set = 0")
        
        # Sắp xếp theo thời gian
        df = df.sort_index()
        
        if df.empty:
            print(f"  ❌ DataFrame rỗng sau khi clean")
            return None
        
        print(f"  ✓ Parse thành công: {len(df)} records")
        print(f"  Date range: {df.index.min()} đến {df.index.max()}")
        print(f"  Columns: {df.columns.tolist()}")
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, symbol: str) -> str:
        """Lưu DataFrame vào CSV"""
        filename = os.path.join(self.output_dir, f"{symbol}_data.csv")
        df.to_csv(filename, encoding='utf-8')
        print(f"  ✓ Đã lưu: {filename}")
        print(f"  File size: {os.path.getsize(filename) / 1024:.2f} KB")
        return filename
    
    def crawl_and_save(self, symbol: str, api_key: str = None, years_back: int = 3, 
                      resolution: str = "1D") -> Optional[str]:
        """
        Crawl và lưu dữ liệu cho một mã cổ phiếu
        
        Returns:
            Đường dẫn file CSV hoặc None nếu lỗi
        """
        # Fetch data
        data = self.fetch_stock_data(symbol, resolution=resolution, years_back=years_back)
        
        if data is None:
            return None
        
        # Parse data
        df = self.parse_chart_data(data, symbol)
        
        if df is None or df.empty:
            return None
        
        # Save to CSV
        filename = self.save_to_csv(df, symbol)
        
        return filename
    
    def crawl_multiple_symbols(self, symbols: list, years_back: int = 3,
                               delay: float = 1.0) -> Dict[str, str]:
        """
        Crawl nhiều mã cổ phiếu
        
        Args:
            symbols: List mã cổ phiếu
            years_back: Số năm lùi lại
            delay: Delay giữa các request (giây)
        
        Returns:
            Dict {symbol: filepath}
        """
        results = {}
        
        print(f"\n{'='*60}")
        print(f"CRAWL {len(symbols)} MÃ CỔ PHIẾU")
        print(f"{'='*60}\n")
        API_KEY = os.getenv('API_KEY') or os.getenv('BEAVERX_API_KEY')
    
        for i, symbol in enumerate(symbols):
            print(f"[{i+1}/{len(symbols)}] Processing {symbol}...")
            
            filename = self.crawl_and_save(symbol, api_key=API_KEY, years_back=years_back)
            
            if filename:
                results[symbol] = filename
            
            # Delay để tránh rate limit
            if i < len(symbols) - 1:
                print(f"  Waiting {delay}s...")
                time.sleep(delay)
        
        print(f"\n{'='*60}")
        print(f"HOÀN THÀNH: {len(results)}/{len(symbols)} mã thành công")
        print(f"{'='*60}")
        
        return results

def read_symbols_from_file(filepath: str) -> list:
    """
    Đọc danh sách symbols từ file
    
    Args:
        filepath: Đường dẫn file (ví dụ: "vn30_symbols.txt")
    
    Returns:
        List các symbols
    """
    symbols = []
    
    try:
        if not os.path.exists(filepath):
            print(f"⚠️ File không tồn tại: {filepath}")
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                symbol = line.strip()
                # Bỏ qua dòng trống và comment
                if symbol and not symbol.startswith('#'):
                    symbols.append(symbol)
        
        print(f"✓ Đọc được {len(symbols)} symbols từ {filepath}")
        return symbols
        
    except Exception as e:
        print(f"❌ Lỗi khi đọc file {filepath}: {e}")
        return []



# Sử dụng
# Sử dụng
if __name__ == "__main__":
    import os
    
    # Cấu hình
    HOST = "https://price.beaverx.ai"
    YEARS_BACK = 5
    
    # Lấy API key từ environment variable
    API_KEY = os.getenv('API_KEY') or os.getenv('BEAVERX_API_KEY')
    
    # Đọc symbols từ file (ưu tiên các file này)
    SYMBOLS = []
    for filepath in ["vn30_symbols.txt", "vn30_symbols.py"]:
        if os.path.exists(filepath):
            SYMBOLS = read_symbols_from_file(filepath)
            if SYMBOLS:
                break
    
    # Nếu không đọc được từ file, dùng danh sách mặc định
    if not SYMBOLS:
        print(f"⚠️ Không đọc được từ file, dùng danh sách mặc định")
        SYMBOLS = ["ACB", "VNM", "HPG", "VIC", "CTG", "FPT"]
    
    print("=" * 60)
    print("VN STOCK CRAWLER")
    print("=" * 60)
    print(f"Host: {HOST}")
    print(f"API Key: {'✓ Có' if API_KEY else '❌ Không có (sẽ bị lỗi 401)'}")
    print(f"Symbols: {len(SYMBOLS)} mã")
    if len(SYMBOLS) <= 10:
        print(f"  {', '.join(SYMBOLS)}")
    else:
        print(f"  {', '.join(SYMBOLS[:10])}... và {len(SYMBOLS)-10} mã khác")
    print(f"Years back: {YEARS_BACK}")
    print("=" * 60)
    print()
    
    # Tạo crawler với API key
    crawler = VNStockCrawler(host=HOST, output_dir="data", api_key=API_KEY)
    
    results = crawler.crawl_multiple_symbols(SYMBOLS, years_back=YEARS_BACK, delay=1.0)
    
    if results:
        print(f"\n{'='*60}")
        print(f"✓ THÀNH CÔNG!")
        print(f"{'='*60}")
        print(f"Đã crawl {len(results)} mã:")
        for symbol, filename in results.items():
            print(f"  - {symbol}: {filename}")
            
        # Verify một file mẫu
        if results:
            sample_symbol = list(results.keys())[0]
            sample_file = results[sample_symbol]
            print(f"\nĐang verify dữ liệu mẫu ({sample_symbol})...")
            df = pd.read_csv(sample_file, index_col=0, parse_dates=True)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")
            print(f"\n  First 5 rows:")
            print(df.head())
    else:
        print(f"\n{'='*60}")
        print(f"❌ THẤT BẠI!")
        print(f"{'='*60}")
        print("Không thể crawl dữ liệu. Vui lòng kiểm tra:")
        print("  - Host URL có đúng không?")
        print("  - API key có đúng không? (export API_KEY='your-key')")
        print("  - Mã cổ phiếu có tồn tại không?")
        print("  - API có hoạt động không?")