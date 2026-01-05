import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils import load_gold_csv, generate_features, select_feature_target

def main():
    BASE_DIR = os.path.dirname(__file__)
    paths = [os.path.join(BASE_DIR, "gold", "data", "sjc.csv"), 
             os.path.join(BASE_DIR, "gold", "data", "world_gold_cafef.csv")]
    
    df = load_gold_csv(paths)
    df = generate_features(df)
    
    # Train/Test split
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    
    X_train, y_train, _ = select_feature_target(train_df)
    X_test, y_test, _ = select_feature_target(test_df)
    
    # HistGradientBoosting: Model mạnh nhất của sklearn cho dữ liệu dạng bảng
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        l2_regularization=1.0 # Chống overfitting
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n" + "="*40)
    print(f"NEW DIRECTIONAL ACCURACY: {acc*100:.2f}%")
    print("="*40)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    model_path = os.path.join(BASE_DIR, "models", "gold_direction_model.joblib")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

if __name__ == "__main__":
    main()
