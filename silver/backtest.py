import pandas as pd
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from utils import load_silver_data, generate_silver_features, get_train_test_split

def silver_walk_forward_test(window_size=150, step=15):
    """
    Backtest 'Cuốn chiếu' cho Bạc (V7).
    """
    print(f"🕵️‍♂️ Đang thực hiện Backtest V7 cho Bạc (Window: {window_size} ngày, Step: {step} ngày)...")
    df = load_silver_data()
    df = generate_silver_features(df)
    
    results = []
    # Bắt đầu từ ngày thứ window_size, dự báo từng đợt step ngày
    for i in range(window_size, len(df) - step, step):
        train_df = df.iloc[:i]
        test_df = df.iloc[i:i+step]
        
        # Lấy features dựa trên V7 logic
        X_train, y_train, _, _, feature_names = get_train_test_split(train_df, test_size=0.01)
        X_test = test_df[feature_names]
        y_test = test_df['target_dir']
        
        # Kiểm tra xem có đủ 2 class để train không
        if len(np.unique(y_train)) < 2:
            print(f"⚠️ Giai đoạn {test_df['date'].iloc[0].date()}: Bỏ qua do dữ liệu học chỉ có 1 nhãn.")
            continue
            
        # V7: Thêm RFE Feature Selection trong mỗi loop backtest để loại bỏ nhiễu
        base_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(base_rf, n_features_to_select=10, step=1)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Train model V7 (Voting)
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=300, max_depth=3, class_weight='balanced', random_state=42)),
            ('hgb', HistGradientBoostingClassifier(max_iter=50, max_depth=2, l2_regularization=50.0, random_state=42)),
            ('svc', make_pipeline(StandardScaler(), SVC(probability=True, kernel='linear', C=0.01, class_weight='balanced', random_state=42)))
        ]
        model = VotingClassifier(estimators=base_models, voting='soft', weights=[1, 1, 2])
        
        model.fit(X_train_selected, y_train)
        preds = model.predict(X_test_selected)
        acc = accuracy_score(y_test, preds)
        
        results.append(acc)
        print(f"📅 Giai đoạn {test_df['date'].iloc[0].date()} -> {test_df['date'].iloc[-1].date()}: Acc = {acc*100:.2f}%")

    avg_acc = np.mean(results)
    print("\n" + "="*50)
    print(f"📊 ĐỘ CHÍNH XÁC TRUNG BÌNH (V7 BACKTEST): {avg_acc*100:.2f}%")
    print("="*50)
    
    if avg_acc > 0.55:
        print("🚀 Hệ thống V7 cho thấy sự cải thiện rõ rệt!")
    elif avg_acc > 0.52:
        print("📈 Hệ thống bắt đầu ổn định.")
    else:
        print("⚠️ Cần tinh chỉnh thêm các đặc trưng vĩ mô.")

if __name__ == "__main__":
    silver_walk_forward_test()
