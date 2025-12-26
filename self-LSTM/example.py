"""
Ví dụ sử dụng nhanh để test models
"""
from compare_models import compare_models

if __name__ == '__main__':
    print("Bắt đầu train và so sánh LSTM vs Transformer...")
    print("Có thể mất vài phút để download dữ liệu và train models...\n")
    
    # So sánh models với cấu hình mặc định
    results = compare_models(
        symbol='AAPL',           # Mã cổ phiếu
        start_date='2020-01-01', # Ngày bắt đầu
        end_date='2024-01-01',   # Ngày kết thúc
        lookback_days=60,        # Số ngày lookback
        forecast_days=3,          # Dự báo 3 ngày
        epochs=50                # Số epochs (có thể giảm xuống 20-30 để test nhanh)
    )
    
    print("\n✓ Hoàn thành! Kiểm tra thư mục 'plots' và 'checkpoints' để xem kết quả.")

