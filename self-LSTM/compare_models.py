"""
Script để so sánh LSTM và Transformer models
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from train import train_model
from data_loader import StockDataLoader


def compare_models(symbol='AAPL', start_date='2020-01-01', end_date='2024-01-01',
                   lookback_days=60, forecast_days=3, epochs=50):
    """
    So sánh LSTM và Transformer models
    
    Args:
        symbol: Mã cổ phiếu
        start_date: Ngày bắt đầu
        end_date: Ngày kết thúc
        lookback_days: Số ngày lookback
        forecast_days: Số ngày dự báo
        epochs: Số epochs để train
    """
    print("=" * 70)
    print("SO SÁNH LSTM VÀ TRANSFORMER MODELS")
    print("=" * 70)
    
    results = {}
    
    # Train LSTM
    print("\n" + "=" * 70)
    print("1. TRAINING LSTM MODEL")
    print("=" * 70)
    lstm_trainer, lstm_preds, lstm_actuals, lstm_metrics = train_model(
        model_type='lstm',
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days,
        forecast_days=forecast_days,
        hidden_size=128,
        num_layers=3,
        epochs=epochs,
        batch_size=32
    )
    results['LSTM'] = {
        'trainer': lstm_trainer,
        'predictions': lstm_preds,
        'actuals': lstm_actuals,
        'metrics': lstm_metrics
    }
    
    # Train Transformer
    print("\n" + "=" * 70)
    print("2. TRAINING TRANSFORMER MODEL")
    print("=" * 70)
    transformer_trainer, transformer_preds, transformer_actuals, transformer_metrics = train_model(
        model_type='transformer',
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days,
        forecast_days=forecast_days,
        hidden_size=128,
        num_layers=4,
        epochs=epochs,
        batch_size=32
    )
    results['Transformer'] = {
        'trainer': transformer_trainer,
        'predictions': transformer_preds,
        'actuals': transformer_actuals,
        'metrics': transformer_metrics
    }
    
    # So sánh metrics
    print("\n" + "=" * 70)
    print("3. SO SÁNH KẾT QUẢ")
    print("=" * 70)
    
    comparison_df = pd.DataFrame({
        'LSTM': lstm_metrics,
        'Transformer': transformer_metrics
    })
    
    print("\nBảng so sánh metrics:")
    print(comparison_df.to_string())
    
    # Xác định model tốt hơn
    print("\n" + "-" * 70)
    print("ĐÁNH GIÁ:")
    print("-" * 70)
    
    metrics_to_compare = ['RMSE', 'MAE', 'MAPE']
    for metric in metrics_to_compare:
        if metric in lstm_metrics and metric in transformer_metrics:
            lstm_val = lstm_metrics[metric]
            trans_val = transformer_metrics[metric]
            
            if lstm_val < trans_val:
                improvement = ((trans_val - lstm_val) / trans_val) * 100
                print(f"{metric}: LSTM tốt hơn {improvement:.2f}% (LSTM: {lstm_val:.4f} vs Transformer: {trans_val:.4f})")
            else:
                improvement = ((lstm_val - trans_val) / lstm_val) * 100
                print(f"{metric}: Transformer tốt hơn {improvement:.2f}% (Transformer: {trans_val:.4f} vs LSTM: {lstm_val:.4f})")
    
    # Vẽ đồ thị so sánh
    plot_comparison(results, symbol, forecast_days)
    
    # Vẽ đồ thị predictions
    plot_predictions(results, symbol, forecast_days)
    
    return results


def plot_comparison(results, symbol, forecast_days):
    """Vẽ đồ thị so sánh loss của 2 models"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training losses
    axes[0].plot(results['LSTM']['trainer'].train_losses, label='LSTM Train', alpha=0.7)
    axes[0].plot(results['LSTM']['trainer'].val_losses, label='LSTM Val', alpha=0.7)
    axes[0].plot(results['Transformer']['trainer'].train_losses, label='Transformer Train', alpha=0.7)
    axes[0].plot(results['Transformer']['trainer'].val_losses, label='Transformer Val', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot metrics comparison
    metrics = ['RMSE', 'MAE', 'MAPE']
    lstm_values = [results['LSTM']['metrics'].get(m, 0) for m in metrics]
    trans_values = [results['Transformer']['metrics'].get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1].bar(x - width/2, lstm_values, width, label='LSTM', alpha=0.8)
    axes[1].bar(x + width/2, trans_values, width, label='Transformer', alpha=0.8)
    axes[1].set_xlabel('Metrics')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Metrics Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend()
    axes[1].grid(True, axis='y')
    
    plt.tight_layout()
    import os
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/comparison_{symbol}.png', dpi=300, bbox_inches='tight')
    print(f"\nĐã lưu đồ thị so sánh: plots/comparison_{symbol}.png")
    plt.close()


def plot_predictions(results, symbol, forecast_days):
    """Vẽ đồ thị predictions vs actuals"""
    fig, axes = plt.subplots(forecast_days, 2, figsize=(15, 5 * forecast_days))
    
    if forecast_days == 1:
        axes = axes.reshape(1, -1)
    
    for day in range(forecast_days):
        # LSTM predictions
        lstm_pred = results['LSTM']['predictions'][:, day] if forecast_days > 1 else results['LSTM']['predictions'].flatten()
        lstm_actual = results['LSTM']['actuals'][:, day] if forecast_days > 1 else results['LSTM']['actuals'].flatten()
        
        axes[day, 0].scatter(lstm_actual, lstm_pred, alpha=0.5)
        axes[day, 0].plot([lstm_actual.min(), lstm_actual.max()], 
                         [lstm_actual.min(), lstm_actual.max()], 'r--', lw=2)
        axes[day, 0].set_xlabel('Actual Price')
        axes[day, 0].set_ylabel('Predicted Price')
        axes[day, 0].set_title(f'LSTM - Day {day+1} Prediction')
        axes[day, 0].grid(True)
        
        # Transformer predictions
        trans_pred = results['Transformer']['predictions'][:, day] if forecast_days > 1 else results['Transformer']['predictions'].flatten()
        trans_actual = results['Transformer']['actuals'][:, day] if forecast_days > 1 else results['Transformer']['actuals'].flatten()
        
        axes[day, 1].scatter(trans_actual, trans_pred, alpha=0.5)
        axes[day, 1].plot([trans_actual.min(), trans_actual.max()], 
                         [trans_actual.min(), trans_actual.max()], 'r--', lw=2)
        axes[day, 1].set_xlabel('Actual Price')
        axes[day, 1].set_ylabel('Predicted Price')
        axes[day, 1].set_title(f'Transformer - Day {day+1} Prediction')
        axes[day, 1].grid(True)
    
    plt.tight_layout()
    import os
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/predictions_{symbol}.png', dpi=300, bbox_inches='tight')
    print(f"Đã lưu đồ thị predictions: plots/predictions_{symbol}.png")
    plt.close()


if __name__ == '__main__':
    # So sánh models
    results = compare_models(
        symbol='AAPL',
        start_date='2020-01-01',
        end_date='2024-01-01',
        lookback_days=60,
        forecast_days=3,
        epochs=50
    )
    
    print("\n" + "=" * 70)
    print("HOÀN THÀNH SO SÁNH!")
    print("=" * 70)

