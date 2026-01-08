import pandas as pd
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from utils_v2 import load_data, generate_advanced_features, get_train_test_split

def walk_forward_validation(df, window_size=100, step_size=20):
    """
    Thực hiện backtest kiểu Walk-forward (cuốn chiếu) để đánh giá độ ổn định của mô hình.
    """
    print(f"🔄 Bắt đầu Backtest Walk-forward (Window: {window_size}, Step: {step_size})...")
    
    results = []
    
    # Bắt đầu từ window_size
    for i in range(window_size, len(df) - step_size, step_size):
        # 1. Dữ liệu training (từ đầu đến i)
        train_df = df.iloc[:i]
        # 2. Dữ liệu testing (từ i đến i + step_size)
        test_df = df.iloc[i:i + step_size]
        
        # Lấy X, y
        # Tái sử dụng logic lấy features từ utils_v2
        from utils_v2 import get_train_test_split
        X_train, y_train, _, _, feature_names = get_train_test_split(train_df, test_size=0.01)
        X_test = test_df[feature_names]
        y_test = test_df['target_dir']
        
        # 3. Train model
        model = HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        # 4. Predict
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        results.append({
            'start_date': test_df['date'].iloc[0].strftime('%Y-%m-%d'),
            'end_date': test_df['date'].iloc[-1].strftime('%Y-%m-%d'),
            'accuracy': acc
        })
        
        print(f"📅 Giai đoạn {results[-1]['start_date']} -> {results[-1]['end_date']}: Acc = {acc*100:.2f}%")
        
    avg_acc = np.mean([r['accuracy'] for r in results])
    print("\n" + "="*40)
    print(f"📊 ĐỘ CHÍNH XÁC TRUNG BÌNH BACKTEST: {avg_acc*100:.2f}%")
    print("="*40)
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = load_data()
    df = generate_advanced_features(df)
    if len(df) > 150:
        walk_forward_validation(df)
    else:
        print("⚠️ Dữ liệu quá ít để thực hiện Walk-forward validation.")

