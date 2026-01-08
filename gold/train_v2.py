import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from utils_v2 import load_data, generate_advanced_features, get_train_test_split

from sklearn.feature_selection import RFE
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier

def train_and_evaluate():
    print("🚀 Bắt đầu huấn luyện mô hình V8 (Feature Selection & Voting)...")
    
    # 1. Load and Prepare Data
    df = load_data()
    df = generate_advanced_features(df)
    
    if df.empty:
        print("❌ Dữ liệu trống sau khi tạo đặc trưng.")
        return

    # 2. Split Data
    X_train, y_train, X_test, y_test, feature_names = get_train_test_split(df)
    
    print(f"📊 Số lượng mẫu training: {len(X_train)}")
    print(f"📊 Số lượng mẫu testing: {len(X_test)}")

    # 3. FEATURE SELECTION: Lọc ra 12 đặc trưng tốt nhất để tránh nhiễu
    # Với 300 mẫu, 12 đặc trưng là con số "vàng" để tránh Overfitting
    selector = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=12)
    selector = selector.fit(X_train, y_train)
    selected_features = [f for f, s in zip(feature_names, selector.support_) if s]
    
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]
    
    print(f"🎯 Đã chọn lọc 12/25 đặc trưng quan trọng nhất.")
    print(f"📊 Các đặc trưng giữ lại: {selected_features}")

    # 4. Định nghĩa các mô hình cơ sở (Base Models) với tham số lỳ lợm
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=15, random_state=42)),
        ('hgb', HistGradientBoostingClassifier(max_iter=200, max_depth=3, l2_regularization=30.0, random_state=42)),
        ('svc', make_pipeline(StandardScaler(), SVC(probability=True, kernel='linear', C=0.1, random_state=42)))
    ]

    # 5. Sử dụng Voting Classifier thay vì Stacking để giảm độ phức tạp
    model = VotingClassifier(
        estimators=base_models,
        voting='soft', 
        weights=[1, 1, 0.8]
    )
    
    model.fit(X_train_sel, y_train)

    # 6. Evaluation
    y_pred = model.predict(X_test_sel)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"🏆 ĐỘ CHÍNH XÁC V8 (Voting + RFE): {acc*100:.2f}%")
    print("="*50)
    print("\nBáo cáo chi tiết:")
    print(classification_report(y_test, y_pred))

    # 7. Save Model
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "gold_v2_classifier.joblib")
    
    joblib.dump({
        'model': model,
        'feature_names': selected_features, # QUAN TRỌNG: Lưu feature đã lọc
        'accuracy': acc,
        'date': pd.Timestamp.now().strftime('%Y-%m-%d')
    }, model_path)
    
    print(f"✅ Đã lưu mô hình tại: {model_path}")

    # 6. Simple Backtest on Test Set
    # Lấy lại phần dữ liệu test từ dataframe gốc để có đầy đủ các cột (như target_ret)
    df_trainable = df.iloc[30:-1]
    test_results = df_trainable.iloc[-len(y_test):].copy()
    test_results['pred_dir'] = y_pred
    
    # Giả định: Nếu đoán đúng 1 hoặc -1 thì có lời, nếu sai thì lỗ
    # Đây là cách tính lợi nhuận đơn giản dựa trên hướng đi
    test_results['profit'] = test_results.apply(
        lambda row: abs(row['target_ret']) if row['pred_dir'] == row['target_dir'] and row['pred_dir'] != 0
        else (-abs(row['target_ret']) if row['pred_dir'] != row['target_dir'] and row['pred_dir'] != 0 else 0),
        axis=1
    )
    
    cumulative_profit = (1 + test_results['profit']).prod() - 1
    print(f"📈 Lợi nhuận tích lũy giả định trên tập Test: {cumulative_profit*100:.2f}%")

if __name__ == "__main__":
    train_and_evaluate()

