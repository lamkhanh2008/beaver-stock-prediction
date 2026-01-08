import pandas as pd
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from utils import load_silver_data, generate_silver_features, get_train_test_split

def silver_walk_forward_test(window_size=120, step=20):
    print(f"🕵️‍♂️ Đang thực hiện kiểm tra chéo 'Cuốn chiếu' cho Bạc (Window: {window_size} ngày)...")
    df = load_silver_data()
    df = generate_silver_features(df)
    
    results = []
    # Bắt đầu từ ngày thứ window_size, dự báo từng đợt step ngày
    for i in range(window_size, len(df) - step, step):
        train_df = df.iloc[:i]
        test_df = df.iloc[i:i+step]
        
        # Lấy features
        X_train, y_train, _, _, feature_names = get_train_test_split(train_df, test_size=0.01)
        X_test = test_df[feature_names]
        y_test = test_df['target_dir']
        
        # Train model (Stacking V5)
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ('hgb', HistGradientBoostingClassifier(max_iter=100, max_depth=3, random_state=42)),
            ('svc', make_pipeline(StandardScaler(), SVC(probability=True, kernel='linear', random_state=42)))
        ]
        model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=3)
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        results.append(acc)
        print(f"📅 Giai đoạn {test_df['date'].iloc[0].date()} -> {test_df['date'].iloc[-1].date()}: Accuracy = {acc*100:.2f}%")

    avg_acc = np.mean(results)
    print("\n" + "="*50)
    print(f"📊 ĐỘ CHÍNH XÁC THỰC TẾ TRUNG BÌNH (BACKTEST): {avg_acc*100:.2f}%")
    print("="*50)
    
    if avg_acc > 0.60:
        print("🚀 Hệ thống rất ổn định và tin cậy cao!")
    elif avg_acc > 0.52:
        print("📈 Hệ thống có lợi thế (Edge) so với ngẫu nhiên, nhưng cần cẩn thận.")
    else:
        print("⚠️ Hệ thống chưa ổn định, không nên tin cậy hoàn toàn.")

if __name__ == "__main__":
    silver_walk_forward_test()


