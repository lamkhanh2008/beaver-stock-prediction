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

from sklearn.feature_selection import RFE

def train_silver_model():
    print("🚀 Đang huấn luyện mô hình Bạc V7 (Advanced Features & RFE Selection)...")
    
    # 1. Load data
    df = load_silver_data()
    df = generate_silver_features(df)
    
    # 2. Split
    X_train, y_train, X_test, y_test, feature_names = get_train_test_split(df)
    
    # 3. Feature Selection bằng RFE (Recursive Feature Elimination)
    # Dùng RandomForest làm estimator cơ sở để chọn lọc
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFE(estimator, n_features_to_select=10, step=1)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
    print(f"🎯 Top 10 đặc trưng mạnh nhất: {selected_features}")

    # 4. Define Base Models với Regularization cực mạnh
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=500, max_depth=3, min_samples_leaf=30, 
                                     class_weight='balanced', random_state=42)),
        ('hgb', HistGradientBoostingClassifier(max_iter=50, max_depth=2, l2_regularization=100.0, 
                                               random_state=42)),
        ('svc', make_pipeline(StandardScaler(), SVC(probability=True, kernel='linear', C=0.01, 
                                                   class_weight='balanced', random_state=42)))
    ]

    # 5. Voting Classifier (Soft voting)
    model = VotingClassifier(
        estimators=base_models,
        voting='soft',
        weights=[1, 1, 2] # Ưu tiên SVC vì ổn định nhất trên dữ liệu nhỏ
    )
    
    print("⏳ AI đang tối ưu hóa các mẫu hình biến động...")
    model.fit(X_train_selected, y_train)

    # 6. Evaluation
    y_pred = model.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"🏆 ĐỘ CHÍNH XÁC DỰ BÁO BẠC (V7): {acc*100:.2f}%")
    print("="*50)

    # 7. Save
    model_dir = "silver/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "silver_v1_model.joblib")
    
    joblib.dump({
        'model': model,
        'selector': selector,
        'feature_names': feature_names,
        'selected_features': selected_features,
        'accuracy': acc,
        'date': pd.Timestamp.now().strftime('%Y-%m-%d')
    }, model_path)
    
    print(f"✅ Mô hình V7 đã được lưu thành công!")

if __name__ == "__main__":
    train_silver_model()


