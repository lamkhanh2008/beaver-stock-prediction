# Đánh giá số Layers cần thiết cho dữ liệu cổ phiếu

## Tổng quan

Dựa trên phân tích dữ liệu cổ phiếu và thực nghiệm, đây là khuyến nghị về số layers cho các models.

## Đặc điểm dữ liệu cổ phiếu

- **Số lượng features**: 50-100+ features (giá, volume, technical indicators, custom features)
- **Tính chất**: Time series với nhiều patterns phức tạp
- **Noise**: Dữ liệu có nhiều noise, cần regularization
- **Temporal dependencies**: Phụ thuộc mạnh vào lịch sử (60+ ngày)

## Khuyến nghị cho LSTM

### 2 Layers
- **Phù hợp**: Dữ liệu đơn giản, ít features (< 30)
- **Ưu điểm**: Train nhanh, ít overfitting
- **Nhược điểm**: Có thể không capture được patterns phức tạp

### 3 Layers ⭐ (KHUYẾN NGHỊ)
- **Phù hợp**: Hầu hết trường hợp với dữ liệu cổ phiếu
- **Ưu điểm**: 
  - Cân bằng tốt giữa độ phức tạp và performance
  - Đủ để học các patterns phức tạp
  - Không quá phức tạp để tránh overfitting
- **Cấu hình**: 
  - Hidden size: 128
  - Dropout: 0.2
  - Lookback: 60 ngày

### 4+ Layers
- **Phù hợp**: Dữ liệu rất phức tạp, nhiều features (> 80)
- **Ưu điểm**: Có thể học được patterns rất phức tạp
- **Nhược điểm**: 
  - Cần nhiều dữ liệu hơn
  - Dễ overfitting
  - Train lâu hơn

## Khuyến nghị cho Transformer

### 2-3 Layers
- **Phù hợp**: Dữ liệu đơn giản đến trung bình
- **Ưu điểm**: Train nhanh, ít parameters
- **Nhược điểm**: Có thể không đủ cho dữ liệu phức tạp

### 4 Layers ⭐ (KHUYẾN NGHỊ)
- **Phù hợp**: Dữ liệu cổ phiếu với nhiều features
- **Ưu điểm**:
  - Transformer cần nhiều layers hơn LSTM để đạt performance tương đương
  - 4 layers là điểm sweet spot
  - Attention mechanism giúp capture dependencies tốt
- **Cấu hình**:
  - d_model: 128
  - nhead: 8
  - num_layers: 4
  - dim_feedforward: 512

### 6+ Layers
- **Phù hợp**: Dữ liệu rất lớn và phức tạp
- **Nhược điểm**: 
  - Cần rất nhiều dữ liệu
  - Có thể quá phức tạp cho bài toán này
  - Train rất lâu

## So sánh LSTM vs Transformer

| Aspect | LSTM (3 layers) | Transformer (4 layers) |
|--------|----------------|----------------------|
| Số parameters | ~200K-500K | ~300K-800K |
| Tốc độ train | Nhanh hơn | Chậm hơn |
| Memory | Ít hơn | Nhiều hơn |
| Performance | Tốt cho sequential data | Tốt cho long-range dependencies |
| Overfitting risk | Trung bình | Cao hơn (cần nhiều dữ liệu) |

## Kết luận

### Cho dữ liệu cổ phiếu tiêu chuẩn:
- **LSTM**: **3 layers** với hidden_size=128
- **Transformer**: **4 layers** với d_model=128

### Lý do:
1. **Đủ phức tạp**: Có thể học được các patterns trong dữ liệu cổ phiếu
2. **Không quá phức tạp**: Tránh overfitting với dữ liệu hạn chế
3. **Cân bằng**: Giữa performance và tốc độ train
4. **Thực nghiệm**: Đã được test và cho kết quả tốt

### Khi nào tăng số layers:
- Có nhiều dữ liệu (> 5 năm, nhiều mã cổ phiếu)
- Nhiều features (> 100)
- Cần độ chính xác cao hơn và sẵn sàng trade-off với tốc độ

### Khi nào giảm số layers:
- Ít dữ liệu (< 2 năm)
- Ít features (< 30)
- Cần train nhanh
- Overfitting rõ ràng

## Thực nghiệm

Để test số layers phù hợp, có thể chạy:

```python
from train import train_model

# Test với 2 layers
train_model(..., num_layers=2, ...)

# Test với 3 layers (khuyến nghị)
train_model(..., num_layers=3, ...)

# Test với 4 layers
train_model(..., num_layers=4, ...)
```

So sánh validation loss và test metrics để chọn số layers tối ưu cho dữ liệu cụ thể của bạn.

