# Stock Price Prediction - LSTM vs Transformer

Dự án định giá chứng khoán sử dụng LSTM và Transformer để dự báo giá cổ phiếu trong 1-3 ngày.

## Tính năng

- **Nhiều input features**: Không chỉ dựa trên time series, model sử dụng nhiều features:
  - Giá (Open, High, Low, Close)
  - Volume
  - Technical indicators (RSI, MACD, Moving Averages, etc.)
  - Custom features (volatility, price ratios, volume ratios)
  
- **LSTM Model**: 
  - Multi-layer LSTM với attention mechanism
  - Hỗ trợ nhiều layers (khuyến nghị: 2-4 layers)
  - Dropout để tránh overfitting

- **Transformer Model**:
  - Transformer encoder với positional encoding
  - Multi-head attention
  - So sánh trực tiếp với LSTM

- **Dự báo 1-3 ngày**: Hỗ trợ dự báo giá đóng cửa cho 1-3 ngày tiếp theo

## Cấu trúc dự án

```
Stock-Evaluation/
├── data_loader.py          # Load và preprocess dữ liệu
├── models/
│   ├── __init__.py
│   ├── lstm_model.py       # LSTM model
│   └── transformer_model.py # Transformer model
├── train.py                # Training script
├── compare_models.py       # So sánh LSTM vs Transformer
├── utils.py                # Utility functions
├── requirements.txt        # Dependencies
└── README.md
```

## Cài đặt

1. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

## Sử dụng

### 1. Train một model riêng lẻ

**Train LSTM:**
```python
from train import train_model

trainer, predictions, actuals, metrics = train_model(
    model_type='lstm',
    symbol='AAPL',
    start_date='2020-01-01',
    end_date='2024-01-01',
    lookback_days=60,
    forecast_days=3,
    hidden_size=128,
    num_layers=3,
    epochs=50,
    batch_size=32
)
```

**Train Transformer:**
```python
trainer, predictions, actuals, metrics = train_model(
    model_type='transformer',
    symbol='AAPL',
    start_date='2020-01-01',
    end_date='2024-01-01',
    lookback_days=60,
    forecast_days=3,
    hidden_size=128,
    num_layers=4,
    epochs=50,
    batch_size=32
)
```

### 2. So sánh LSTM và Transformer

```bash
python compare_models.py
```

Hoặc trong code:
```python
from compare_models import compare_models

results = compare_models(
    symbol='AAPL',
    start_date='2020-01-01',
    end_date='2024-01-01',
    lookback_days=60,
    forecast_days=3,
    epochs=50
)
```

## Đánh giá số layers cần thiết

Dựa trên dữ liệu cổ phiếu và thực nghiệm:

### LSTM:
- **2 layers**: Đủ cho dữ liệu đơn giản, ít features
- **3 layers** (khuyến nghị): Cân bằng tốt giữa độ phức tạp và performance
- **4+ layers**: Có thể overfitting nếu dữ liệu ít, nhưng tốt cho dữ liệu phức tạp

### Transformer:
- **2-3 layers**: Đủ cho hầu hết trường hợp
- **4 layers** (khuyến nghị): Tốt cho dữ liệu phức tạp với nhiều features
- **6+ layers**: Có thể quá phức tạp, cần nhiều dữ liệu

### Khuyến nghị:
- **Lookback days**: 60 ngày (đủ để capture patterns)
- **LSTM layers**: 3 layers
- **Transformer layers**: 4 layers
- **Hidden size**: 128 (cân bằng performance và tốc độ)

## Metrics đánh giá

- **MSE** (Mean Squared Error): Lỗi bình phương trung bình
- **MAE** (Mean Absolute Error): Lỗi tuyệt đối trung bình
- **RMSE** (Root Mean Squared Error): Căn bậc hai của MSE
- **MAPE** (Mean Absolute Percentage Error): Phần trăm lỗi trung bình
- **Direction Accuracy**: Độ chính xác dự đoán hướng (tăng/giảm)

## Output

Sau khi train, các file sau sẽ được tạo:

- `checkpoints/`: Chứa model weights tốt nhất
- `plots/`: Chứa các đồ thị:
  - `{model_name}_losses.png`: Training và validation loss
  - `comparison_{symbol}.png`: So sánh giữa LSTM và Transformer
  - `predictions_{symbol}.png`: Predictions vs Actuals

## Ví dụ sử dụng

```python
from data_loader import StockDataLoader
from models.lstm_model import StockLSTM
import torch

# Load data
data_loader = StockDataLoader(
    symbol='AAPL',
    start_date='2020-01-01',
    end_date='2024-01-01',
    lookback_days=60,
    forecast_days=3
)

X_train, X_test, y_train, y_test = data_loader.get_train_test_split()

# Tạo model
input_size = X_train.shape[2]
model = StockLSTM(
    input_size=input_size,
    hidden_size=128,
    num_layers=3,
    forecast_days=3
)

print(f"Số features: {input_size}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
```

## Lưu ý

1. **Dữ liệu**: Script tự động download từ Yahoo Finance, có thể cần VPN nếu ở Việt Nam
2. **GPU**: Nếu có GPU, model sẽ tự động sử dụng để train nhanh hơn
3. **Early stopping**: Model sẽ tự động dừng nếu validation loss không cải thiện
4. **Features**: Số lượng features có thể thay đổi tùy theo dữ liệu, thường từ 50-100+ features

## Tùy chỉnh

### Thêm features mới:
Chỉnh sửa hàm `add_custom_features()` trong `data_loader.py`

### Thay đổi kiến trúc model:
Chỉnh sửa các file trong thư mục `models/`

### Thay đổi hyperparameters:
Sửa các tham số trong `train.py` hoặc `compare_models.py`

## Kết quả mong đợi

- **MAPE**: Thường từ 1-5% cho dự báo 1 ngày, 2-8% cho 3 ngày
- **Direction Accuracy**: Thường từ 55-65% (tốt hơn random 50%)
- **RMSE**: Phụ thuộc vào giá cổ phiếu (thường < 5% giá trị trung bình)

## Tác giả

Dự án nghiên cứu định giá chứng khoán sử dụng Deep Learning

