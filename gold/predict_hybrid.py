import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from utils_v2 import load_data, generate_advanced_features

# --- CÁC ĐỢT CAN THIỆP CỦA NHNN (SBV) ---
SBV_INTERVENTION_DATES = ["2024-05-20", "2024-06-03", "2024-10-15", "2025-01-02"]

def rule_based_expert(row):
    """
    Expert system logic with enhanced rules for higher reliability.
    Returns a score. Positive for UP, Negative for DOWN.
    """
    score = 0
    # 1. World Return effect (Strong Correlation)
    score += row['xau_ret_1'] * 250
    
    # 2. RSI Extremes (Overbought/Oversold)
    if row['xau_rsi'] > 75: score -= 15 # Overbought pressure
    elif row['xau_rsi'] < 25: score += 15 # Oversold support
    
    # 3. Basis Z-Score effect (Mean Reversion)
    # If SJC is > 10M expensive vs world (z-score high), expect a drop
    score += row['basis_zscore'] * (-20)
    
    # 4. Bollinger Bands Positioning
    if row['xau_bb_pos'] > 0.9: score -= 10
    elif row['xau_bb_pos'] < 0.1: score += 10
    
    # 5. SBV Intervention effect
    is_sbv = 1 if row['date'].strftime('%Y-%m-%d') in SBV_INTERVENTION_DATES else 0
    if is_sbv:
        score -= 30 # Stronger penalty for intervention days
        
    # 6. Trend Momentum
    score += row['sjc_ret_1'] * 40
    
    return score

def predict_hybrid():
    BASE_DIR = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(BASE_DIR, "models", "gold_v2_classifier.joblib")
    
    # 1. Load latest data
    df = load_data()
    df = generate_advanced_features(df)
    last_row = df.iloc[-1]
    
    print(f"🕒 Ngày dự báo: {last_row['date'].strftime('%Y-%m-%d')}")
    print(f"💰 Giá SJC hiện tại: {last_row['sjc_price']} triệu VND/lượng")
    
    # 2. Expert System Prediction
    expert_score = rule_based_expert(last_row)
    expert_dir = 1 if expert_score > 0.08 else (-1 if expert_score < -0.08 else 0)
    
    # 3. ML Model Prediction
    ml_dir = 0
    ml_prob = [0.33, 0.33, 0.33]
    
    if os.path.exists(MODEL_PATH):
        try:
            bundle = joblib.load(MODEL_PATH)
            model = bundle['model']
            feature_names = bundle['feature_names']
            
            X = last_row[feature_names].values.reshape(1, -1)
            ml_dir = model.predict(X)[0]
            ml_prob = model.predict_proba(X)[0]
        except Exception as e:
            print(f"⚠️ Không thể tải mô hình ML: {e}")
    else:
        print("⚠️ Không tìm thấy file mô hình ML. Chỉ sử dụng Expert System.")

    # 4. Hybrid Logic (Weighted Ensemble)
    # ML weight: 0.6, Expert weight: 0.4
    ml_conf = np.max(ml_prob)
    expert_conf = min(abs(expert_score) * 2, 1.0) # Normalize score to 0-1
    
    # Voting logic
    weighted_score = (ml_dir * ml_conf * 0.6) + (expert_dir * expert_conf * 0.4)
    
    if weighted_score > 0.15:
        final_dir = 1
    elif weighted_score < -0.15:
        final_dir = -1
    else:
        final_dir = 0

    confidence_score = (ml_conf * 0.6 + expert_conf * 0.4) * 100
    
    if confidence_score > 75:
        confidence_text = "Rất Cao"
    elif confidence_score > 60:
        confidence_text = "Cao"
    elif confidence_score > 45:
        confidence_text = "Trung bình"
    else:
        confidence_text = "Thấp"

    # 5. Output Results
    res_map = {1: "TĂNG 📈", -1: "GIẢM 📉", 0: "ĐI NGANG ➡️"}
    
    print("\n" + " 🏆 KẾT QUẢ DỰ BÁO HYBRID V3 🏆 ".center(50, "✨"))
    print(f"🔹 Expert Score: {expert_score:.4f} (Conf: {expert_conf*100:.1f}%)")
    print(f"🔹 ML Prob:     {ml_conf*100:.1f}% (Direction: {res_map[ml_dir]})")
    print("-" * 50)
    print(f"🔮 DỰ BÁO CUỐI CÙNG: {res_map[final_dir]}")
    print(f"🛡️ ĐỘ TIN CẬY:       {confidence_text} ({confidence_score:.1f}%)")
    print("-" * 50)
    
    # Ước tính mức giá (Sử dụng trung bình biến động 0.5%)
    avg_move = 0.005
    pred_price = last_row['sjc_price'] * (1 + (final_dir * avg_move))
    if final_dir != 0:
        print(f"🎯 Giá mục tiêu ngày mai: ~{pred_price:.2f} triệu VND/lượng")

if __name__ == "__main__":
    predict_hybrid()

