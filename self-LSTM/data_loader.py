"""
Data Loader cho bài toán định giá chứng khoán
Hỗ trợ load và preprocess dữ liệu với nhiều features
"""
import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ta import add_all_ta_features
from ta.utils import dropna
import warnings
warnings.filterwarnings('ignore')


def merge_local_csvs(data_dir='data', output_file='data/all_symbols.csv'):
    """Gộp các file *_data.csv thành một file duy nhất có cột Symbol."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {data_dir}")

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_data.csv')]
    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy file *_data.csv trong {data_dir}")

    frames = []
    for file in csv_files:
        sym = file.replace('_data.csv', '')
        path = os.path.join(data_dir, file)
        df = pd.read_csv(path)
        if df.empty:
            continue

        date_col = df.columns[0]
        df = df.rename(columns={date_col: 'Date'})
        df['Symbol'] = sym
        frames.append(df)

    if not frames:
        raise ValueError("Không có dữ liệu hợp lệ để gộp.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_file, index=False)
    return output_file


class StockDataLoader:
    """Class để load và preprocess dữ liệu cổ phiếu"""
    
    def __init__(self, symbol, start_date, end_date, lookback_days=60, forecast_days=3,
                 data_dir='data', data_file=None, use_local_data=False, symbols=None,
                 use_ohlcv_only=True):
        """
        Args:
            symbol: Mã cổ phiếu (ví dụ: 'AAPL', 'MSFT'); đặt 'ALL' để đọc toàn bộ csv trong data_dir
            start_date: Ngày bắt đầu (format: 'YYYY-MM-DD')
            end_date: Ngày kết thúc (format: 'YYYY-MM-DD')
            lookback_days: Số ngày quá khứ để dự đoán (default: 60)
            forecast_days: Số ngày dự báo (1-3 ngày)
            data_dir: Thư mục chứa file csv local
            data_file: Đường dẫn file csv gộp (nếu có sẽ ưu tiên đọc file này)
            use_local_data: True nếu muốn đọc dữ liệu từ data_dir thay vì Yahoo Finance
            symbols: Danh sách mã cần đọc (ưu tiên hơn symbol nếu được cung cấp)
            use_ohlcv_only: True để chỉ dùng 5 cột OHLCV làm input
        """
        self.symbol = symbol
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_days = lookback_days
        self.forecast_days = forecast_days
        self.data_dir = data_dir
        self.data_file = data_file
        self.use_local_data = use_local_data
        self.use_ohlcv_only = use_ohlcv_only
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        self.raw_data = None
        self.processed_data = None
    
    def download_data(self):
        """Download dữ liệu từ Yahoo Finance"""
        print(f"Đang download dữ liệu cho {self.symbol}...")
        ticker = yf.Ticker(self.symbol)
        self.raw_data = ticker.history(start=self.start_date, end=self.end_date)
        
        if self.raw_data.empty:
            raise ValueError(f"Không thể download dữ liệu cho {self.symbol}")
        
        print(f"Download thành công: {len(self.raw_data)} ngày")
        return self.raw_data

    def load_local_data(self):
        """Đọc dữ liệu từ file gộp hoặc thư mục data_dir."""
        if self.data_file:
            return self._load_combined_file()

        return self._load_data_dir()

    def _load_combined_file(self):
        """Đọc dữ liệu từ một file csv đã gộp và có cột Symbol."""
        if not os.path.isfile(self.data_file):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {self.data_file}")

        df = pd.read_csv(self.data_file)
        if df.empty:
            raise ValueError("File dữ liệu rỗng.")

        if 'Symbol' not in df.columns:
            raise ValueError("File dữ liệu cần có cột 'Symbol'.")

        date_col = 'Date' if 'Date' in df.columns else df.columns[0]
        df = df.rename(columns={date_col: 'Date'})
        df['Date'] = pd.to_datetime(df['Date'])

        symbol_filter = set()
        if self.symbol and self.symbol != 'ALL':
            symbol_filter.add(self.symbol)
        if self.symbols:
            symbol_filter.update(self.symbols)
        if symbol_filter:
            df = df[df['Symbol'].isin(symbol_filter)]

        if self.start_date:
            df = df[df['Date'] >= pd.to_datetime(self.start_date)]
        if self.end_date:
            df = df[df['Date'] <= pd.to_datetime(self.end_date)]

        df = df.sort_values('Date').set_index('Date')
        self.raw_data = df
        return self.raw_data

    def _load_data_dir(self):
        """Đọc dữ liệu từ thư mục data_dir. Nếu symbol='ALL' thì gộp tất cả file _data.csv"""
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {self.data_dir}")
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('_data.csv')]
        if not csv_files:
            raise FileNotFoundError(f"Không tìm thấy file *_data.csv trong {self.data_dir}")
        
        # Xác định danh sách symbol cần đọc
        symbol_filter = set()
        if self.symbol and self.symbol != 'ALL':
            symbol_filter.add(self.symbol)
        if self.symbols:
            symbol_filter.update(self.symbols)
        
        data_frames = []
        for file in csv_files:
            sym = file.replace('_data.csv', '')
            if symbol_filter and sym not in symbol_filter:
                continue
            
            path = os.path.join(self.data_dir, file)
            df = pd.read_csv(path)
            if df.empty:
                continue
            
            # Xử lý cột ngày (thường là cột đầu tiên không tên)
            date_col = df.columns[0]
            df = df.rename(columns={date_col: 'Date'})
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').set_index('Date')
            
            # Lọc theo khoảng thời gian nếu có
            if self.start_date:
                df = df[df.index >= pd.to_datetime(self.start_date)]
            if self.end_date:
                df = df[df.index <= pd.to_datetime(self.end_date)]
            
            df['Symbol'] = sym
            data_frames.append(df)
        
        if not data_frames:
            raise ValueError("Không tìm thấy dữ liệu local phù hợp với yêu cầu.")
        
        combined = pd.concat(data_frames)
        combined = combined.sort_index()
        self.raw_data = combined
        return self.raw_data

    def _select_ohlcv(self, df):
        """Chỉ giữ 5 cột OHLCV"""
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Thiếu cột bắt buộc: {missing}")
        return df[required_cols].copy()
    
    def add_technical_indicators(self, df):
        """Thêm các chỉ báo kỹ thuật"""
        try:
            # Reset index để có cột Date
            df = df.reset_index()
            if 'Date' not in df.columns and 'Datetime' in df.columns:
                df['Date'] = df['Datetime']
            
            # Thêm các chỉ báo kỹ thuật
            df = add_all_ta_features(
                df, 
                open="Open", 
                high="High", 
                low="Low", 
                close="Close", 
                volume="Volume",
                fillna=True
            )
            
            # Loại bỏ các cột không cần thiết
            df = df.drop(columns=['Date'], errors='ignore')
            df = df.drop(columns=['Datetime'], errors='ignore')
            
        except Exception as e:
            print(f"Lỗi khi thêm technical indicators: {e}")
            # Nếu lỗi, chỉ dùng các features cơ bản
            pass
        
        return df
    
    def add_custom_features(self, df):
        """Thêm các features tùy chỉnh"""
        # Price features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        
        # Volume features
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Price relative to MA
        df['Price_MA5_Ratio'] = df['Close'] / df['MA_5']
        df['Price_MA20_Ratio'] = df['Close'] / df['MA_20']
        
        # RSI (nếu chưa có từ ta library)
        if 'momentum_rsi' not in df.columns:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (nếu chưa có)
        if 'trend_macd' not in df.columns:
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Fill NaN values
        df = df.bfill().ffill()
        
        return df
    
    def prepare_data(self):
        """Chuẩn bị dữ liệu cho training"""
        if self.use_local_data:
            if self.raw_data is None:
                self.load_local_data()
            return self._prepare_local_data()
        else:
            if self.raw_data is None:
                self.download_data()
            
            # Copy dữ liệu
            df = self.raw_data.copy()

            if self.use_ohlcv_only:
                df = self._select_ohlcv(df)
                df = df.bfill().ffill().fillna(0)

                self.feature_names = ['Open', 'High', 'Low', 'Volume']
                feature_data = df[self.feature_names]
                scaled_features = self.scaler.fit_transform(feature_data)

                close_prices = df['Close'].values.reshape(-1, 1)
                scaled_close = self.price_scaler.fit_transform(close_prices)

                self.processed_data = np.column_stack([scaled_features, scaled_close.flatten()])
            else:
                # Thêm technical indicators
                df = self.add_technical_indicators(df)

                # Thêm custom features
                df = self.add_custom_features(df)

                # Loại bỏ các cột không phải số
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                df = df[numeric_cols]

                # Loại bỏ các cột có quá nhiều NaN
                df = df.dropna(axis=1, thresh=len(df) * 0.5)

                # Fill remaining NaN
                df = df.bfill().ffill()
                df = df.fillna(0)

                # Lưu tên các features
                self.feature_names = [col for col in df.columns if col != 'Close']

                # Scale features (trừ Close)
                feature_data = df.drop(columns=['Close'], errors='ignore')
                scaled_features = self.scaler.fit_transform(feature_data)

                # Scale Close price riêng
                close_prices = df['Close'].values.reshape(-1, 1)
                scaled_close = self.price_scaler.fit_transform(close_prices)

                # Combine
                self.processed_data = np.column_stack([scaled_features, scaled_close.flatten()])
            
            print(f"Số features: {len(self.feature_names) + 1}")
            print(f"Kích thước dữ liệu đã xử lý: {self.processed_data.shape}")
            
            return self.processed_data

    def _prepare_single_symbol(self, df):
        """Xử lý feature cho một mã cổ phiếu"""
        df = df.copy()
        if self.use_ohlcv_only:
            df = self._select_ohlcv(df)
            df = df.bfill().ffill().fillna(0)
            return df

        df = self.add_technical_indicators(df)
        df = self.add_custom_features(df)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = df[numeric_cols]
        
        df = df.dropna(axis=1, thresh=len(df) * 0.5)
        df = df.bfill().ffill()
        df = df.fillna(0)
        return df

    def _prepare_local_data(self):
        """Chuẩn bị dữ liệu khi đọc từ data_dir và gộp nhiều mã"""
        processed_frames = []
        symbols_seen = []
        
        for sym, df_sym in self.raw_data.groupby('Symbol', sort=False):
            df_processed = self._prepare_single_symbol(df_sym)
            df_processed['Symbol'] = sym
            processed_frames.append(df_processed)
            symbols_seen.append(sym)
        
        if not processed_frames:
            raise ValueError("Không có dữ liệu hợp lệ sau khi preprocess.")
        
        combined = pd.concat(processed_frames)
        combined = combined.sort_index()
        combined = combined.fillna(0)
        
        if self.use_ohlcv_only:
            self.feature_names = ['Open', 'High', 'Low', 'Volume']
        else:
            # Ghi nhận danh sách features (bỏ Close và Symbol)
            self.feature_names = [col for col in combined.columns if col not in ['Close', 'Symbol']]

        # Scale features và giá đóng cửa trên toàn bộ tập
        feature_data = combined[self.feature_names]
        scaled_features = self.scaler.fit_transform(feature_data)
        
        close_prices = combined['Close'].values.reshape(-1, 1)
        scaled_close = self.price_scaler.fit_transform(close_prices)
        
        scaled_combined = np.column_stack([scaled_features, scaled_close.flatten()])
        scaled_df = pd.DataFrame(
            scaled_combined,
            index=combined.index,
            columns=self.feature_names + ['Close']
        )
        scaled_df['Symbol'] = combined['Symbol'].values
        
        self.processed_data = scaled_df
        
        print(f"Số features: {len(self.feature_names) + 1}")
        print(f"Tổng số dòng sau khi gộp: {len(scaled_df)} (symbols: {sorted(set(symbols_seen))})")
        return self.processed_data
    
    def _append_sequences_from_array(self, array_data, X, y):
        """Helper: tạo sequence từ numpy array"""
        if len(array_data) < self.lookback_days + self.forecast_days:
            return
        for i in range(self.lookback_days, len(array_data) - self.forecast_days + 1):
            X.append(array_data[i - self.lookback_days:i])
            target_idx = i + self.forecast_days - 1
            y.append(array_data[i:target_idx + 1, -1])

    def _make_sequences_from_array(self, array_data):
        """Tạo X, y từ numpy array"""
        X, y = [], []
        self._append_sequences_from_array(array_data, X, y)
        return np.array(X), np.array(y)

    def create_sequences(self, data):
        """
        Tạo sequences cho LSTM/Transformer
        Returns: X, y
        - X: (samples, lookback_days, features)
        - y: (samples, forecast_days) - giá đóng cửa cho 1-3 ngày tiếp theo
        """
        X, y = [], []
        
        if isinstance(data, pd.DataFrame):
            if 'Symbol' in data.columns:
                for _, group in data.groupby('Symbol', sort=False):
                    array_data = group.drop(columns=['Symbol']).values
                    self._append_sequences_from_array(array_data, X, y)
            else:
                self._append_sequences_from_array(data.values, X, y)
        else:
            self._append_sequences_from_array(data, X, y)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    def get_train_test_split(self, test_size=0.2):
        """Chia train/test set"""
        if self.processed_data is None:
            self.prepare_data()

        if isinstance(self.processed_data, pd.DataFrame) and 'Symbol' in self.processed_data.columns:
            X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
            for _, group in self.processed_data.groupby('Symbol', sort=False):
                array_data = group.drop(columns=['Symbol']).values
                X_sym, y_sym = self._make_sequences_from_array(array_data)
                if len(X_sym) == 0:
                    continue
                split_idx = int(len(X_sym) * (1 - test_size))
                X_train_list.append(X_sym[:split_idx])
                X_test_list.append(X_sym[split_idx:])
                y_train_list.append(y_sym[:split_idx])
                y_test_list.append(y_sym[split_idx:])

            if not X_train_list:
                raise ValueError("Không đủ dữ liệu để tạo train/test split.")

            X_train = np.concatenate(X_train_list, axis=0)
            X_test = np.concatenate(X_test_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)
            y_test = np.concatenate(y_test_list, axis=0)
        else:
            X, y = self.create_sequences(self.processed_data)
            # Split theo thời gian (không shuffle)
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform_price(self, scaled_prices):
        """Chuyển đổi giá đã scale về giá thực"""
        return self.price_scaler.inverse_transform(scaled_prices.reshape(-1, 1)).flatten()
    
    def get_feature_count(self):
        """Trả về số lượng features"""
        if self.processed_data is None:
            self.prepare_data()
        if isinstance(self.processed_data, pd.DataFrame):
            return self.processed_data.drop(columns=['Symbol'], errors='ignore').shape[1]
        return self.processed_data.shape[1]

    def get_latest_sequences(self):
        """
        Lấy sequence lookback mới nhất cho từng mã (dùng để dự báo ngày mai)
        Returns: dict: {symbol: np.array shape (lookback_days, features)}
        """
        if self.processed_data is None:
            self.prepare_data()
        
        sequences = {}
        if isinstance(self.processed_data, pd.DataFrame) and 'Symbol' in self.processed_data.columns:
            for sym, group in self.processed_data.groupby('Symbol'):
                arr = group.drop(columns=['Symbol']).values
                if len(arr) >= self.lookback_days:
                    sequences[sym] = arr[-self.lookback_days:]
        else:
            arr = self.processed_data
            if len(arr) >= self.lookback_days:
                sequences[self.symbol] = arr[-self.lookback_days:]
        return sequences
