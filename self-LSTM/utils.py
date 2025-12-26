"""
Utility functions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_metrics(y_true, y_pred):
    """
    Tính các metrics đánh giá
    
    Args:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
    
    Returns:
        dict: Dictionary chứa các metrics
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


def plot_time_series(actual, predicted, title='Time Series Prediction'):
    """Vẽ đồ thị time series"""
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', alpha=0.7)
    plt.plot(predicted, label='Predicted', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

