import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from utils import load_silver_data, generate_silver_features, get_train_test_split

from sklearn.ensemble import VotingClassifier

from sklearn.utils.class_weight import compute_sample_weight

def train_silver_model():
    print("🚀 Đang huấn luyện mô hình Bạc V6 (Final Optimized Voting)...")
    
    # 1. Load data
    df = load_silver_data()
    df = generate_silver_features(df)
    
    # 2. Split
    X_train, y_train, X_test, y_test, feature_names = get_train_test_split(df)
    
    # 3. Define Base Models with tuned parameters for small data
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=4, class_weight='balanced', random_state=42)),
        ('hgb', HistGradientBoostingClassifier(max_iter=50, max_depth=3, l2_regularization=20.0, random_state=42)),
        ('svc', make_pipeline(StandardScaler(), SVC(probability=True, kernel='rbf', C=1.0, class_weight='balanced', random_state=42)))
    ]

    # 4. Voting Classifier
    model = VotingClassifier(
        estimators=base_models,
        voting='soft',
        weights=[1, 1, 1] 
    )
    
    print("⏳ AI đang đồng bộ hóa các chỉ báo kỹ thuật và vĩ mô...")
    model.fit(X_train, y_train)

    # 5. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"🏆 ĐỘ CHÍNH XÁC DỰ BÁO BẠC (V6): {acc*100:.2f}%")
    print("="*50)

    # 6. Save
    model_dir = "silver/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "silver_v1_model.joblib")
    
    joblib.dump({
        'model': model,
        'feature_names': feature_names,
        'accuracy': acc,
        'date': pd.Timestamp.now().strftime('%Y-%m-%d')
    }, model_path)
    
    print(f"✅ Mô hình V6 đã được lưu thành công!")

if __name__ == "__main__":
    train_silver_model()


