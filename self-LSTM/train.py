"""
Training script cho LSTM và Transformer models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from data_loader import StockDataLoader
from models.lstm_model import StockLSTM
from models.transformer_model import StockTransformer


class Trainer:
    """Class để train models"""
    
    def __init__(self, model, device, model_name='model'):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.train_losses = []
        self.val_losses = []
        
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, patience=10,
              loss_type='huber', weight_decay=1e-4):
        """Train model"""
        if loss_type == 'huber':
            criterion = nn.SmoothL1Loss()
        elif loss_type == 'mse':
            criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n=== Training {self.model_name} ===")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {lr}")
        print(f"Loss: {loss_type}, Weight decay: {weight_decay}\n")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(self.model.state_dict(), f'checkpoints/{self.model_name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(f'checkpoints/{self.model_name}_best.pth'))
        print(f"\nBest validation loss: {best_val_loss:.6f}")
        
    def evaluate(self, test_loader, data_loader):
        """Evaluate model trên test set"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                
                # Chuyển về giá thực
                pred_prices = data_loader.inverse_transform_price(outputs.cpu().numpy())
                actual_prices = data_loader.inverse_transform_price(batch_y.numpy())
                
                predictions.extend(pred_prices)
                actuals.extend(actual_prices)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if actuals.ndim == 1:
            actuals = actuals.reshape(-1, 1)
        # Tính metrics
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
        
        # Direction accuracy (độ chính xác hướng)
        if predictions.shape[1] > 1:
            # So sánh hướng thay đổi
            pred_direction = np.diff(predictions, axis=1) > 0
            actual_direction = np.diff(actuals, axis=1) > 0
            direction_accuracy = np.mean(pred_direction == actual_direction) * 100
        else:
            direction_accuracy = None
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy
        }
        
        return predictions, actuals, metrics
    
    def plot_losses(self):
        """Vẽ đồ thị loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.model_name} - Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{self.model_name}_losses.png')
        plt.close()


def train_model(model_type='lstm', symbol='AAPL', start_date='2020-01-01', 
                end_date='2024-01-01', lookback_days=60, forecast_days=1,
                hidden_size=128, num_layers=3, epochs=100, batch_size=32,
                use_local_data=False, data_dir='data', data_file=None, symbols=None, predict_next_day=True,
                use_ohlcv_only=True, loss_type='huber', weight_decay=1e-4, show_sample=True):
    """
    Train một model
    
    Args:
        model_type: 'lstm' hoặc 'transformer'
        symbol: Mã cổ phiếu
        start_date: Ngày bắt đầu
        end_date: Ngày kết thúc
        lookback_days: Số ngày lookback
        forecast_days: Số ngày dự báo (nên để 1 nếu muốn dự báo ngày kế tiếp)
        hidden_size: Kích thước hidden layer
        num_layers: Số layers
        epochs: Số epochs
        batch_size: Batch size
        use_local_data: True để đọc dữ liệu từ data_dir thay vì Yahoo Finance
        data_dir: Thư mục chứa các file *_data.csv
        data_file: File csv gộp có cột Symbol (ưu tiên hơn data_dir nếu có)
        symbols: Danh sách mã cần dùng (nếu None và symbol='ALL' sẽ lấy toàn bộ)
        predict_next_day: Tính toán và in ra giá dự báo cho ngày mai sau khi train
        use_ohlcv_only: True để chỉ dùng 5 cột OHLCV làm input
        loss_type: 'huber' hoặc 'mse'
        weight_decay: L2 regularization cho optimizer
        show_sample: In ra một phần X, danh sách feature, và y mẫu
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_loader = StockDataLoader(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days,
        forecast_days=forecast_days,
        data_dir=data_dir,
        data_file=data_file,
        use_local_data=use_local_data,
        symbols=symbols,
        use_ohlcv_only=use_ohlcv_only
    )
    
    X_train, X_test, y_train, y_test = data_loader.get_train_test_split(test_size=0.2)

    if show_sample:
        input_feature_names = data_loader.feature_names + ['Close']
        print("X_train shape:", X_train.shape)
        print("X_train dtype:", X_train.dtype)
        print("X_train[0] shape:", X_train[0].shape)
        print("Input features (order):", input_feature_names)
        print("X_train[0] (first 2 timesteps):")
        print(X_train[0][:2])  # chỉ in 2 timestep để đỡ dài
        if len(X_train[0]) > 0:
            first_step = X_train[0][0]
            feature_preview = {name: float(val) for name, val in zip(input_feature_names, first_step)}
            print("X_train[0][0] as dict:", feature_preview)
        print("y_train[0]:", y_train[0])
    input_size = X_train.shape[2]
    
    print(f"\nInput size (số features): {input_size}")
    print(f"Lookback days: {lookback_days}")
    print(f"Forecast days: {forecast_days}")
    
    if model_type == 'lstm':
        model = StockLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            forecast_days=forecast_days
        )
    elif model_type == 'transformer':
        model = StockTransformer(
            input_size=input_size,
            d_model=hidden_size,
            num_layers=num_layers,
            forecast_days=forecast_days
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Split train into train/val
    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    # Train
    trainer = Trainer(model, device, model_name=f'{model_type}_{symbol}')
    trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        loss_type=loss_type,
        weight_decay=weight_decay
    )
    trainer.plot_losses()
    
    # Evaluate
    predictions, actuals, metrics = trainer.evaluate(test_loader, data_loader)
    
    print(f"\n=== Evaluation Results for {model_type.upper()} ===")
    for metric_name, metric_value in metrics.items():
        if metric_value is not None:
            print(f"{metric_name}: {metric_value:.4f}")
    
    # Dự báo giá đóng cửa ngày mai (lấy ngày đầu tiên trong forecast_days)
    next_day_predictions = None
    if predict_next_day:
        latest_sequences = data_loader.get_latest_sequences()
        if latest_sequences:
            next_day_predictions = {}
            model.eval()
            for sym, seq in latest_sequences.items():
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
                with torch.no_grad():
                    scaled_forecast = model(seq_tensor).cpu().numpy().flatten()
                forecast_prices = data_loader.inverse_transform_price(scaled_forecast)
                next_day_predictions[sym] = forecast_prices[0]
            
            metrics['Next_Day_Price'] = next_day_predictions
            print("\nDự báo giá đóng cửa ngày mai:")
            for sym, price in next_day_predictions.items():
                print(f"- {sym}: {price:.2f}")
        else:
            print("\nKhông đủ dữ liệu để tạo sequence lookback cho dự báo ngày mai.")
    
    return trainer, predictions, actuals, metrics


if __name__ == '__main__':
    # Train LSTM
    print("=" * 50)
    print("Training LSTM Model")
    print("=" * 50)
    train_model(
        model_type='lstm',
        symbol='ALL',  # Đọc toàn bộ dữ liệu từ file gộp
        start_date='2020-01-01',
        end_date='2024-01-01',
        lookback_days=60,
        forecast_days=1,  # Dự báo 1 ngày (ngày mai)
        hidden_size=128,
        num_layers=3,    # 3 layers LSTM theo yêu cầu
        epochs=50,
        batch_size=32,
        use_local_data=True,
        data_file='data/all_symbols.csv',
        predict_next_day=True
    )
