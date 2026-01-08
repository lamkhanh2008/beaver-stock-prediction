import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from utils import load_silver_data, generate_silver_features

def rule_based_silver_expert(row):
    """Hệ chuyên gia dựa trên quy luật kinh tế của Bạc (Cập nhật V2)"""
    score = 0
    # 1. World Silver Return
    score += row['ag_ret_1'] * 100
    
    # 2. Gold-Silver Ratio (GSR)
    if row['gsr_dist_ma10'] > 0.03:
        score += 15 # GSR cao -> Bạc rẻ -> Kỳ vọng tăng
    elif row['gsr_dist_ma10'] < -0.03:
        score -= 15
        
    # 3. RSI Overbought/Oversold
    if row['ag_rsi'] > 70: score -= 10
    elif row['ag_rsi'] < 30: score += 10
    
    # 4. Khoảng cách MA20 (Mean Reversion)
    if row['dist_ma20'] > 0.05: score -= 10
    elif row['dist_ma20'] < -0.05: score += 10
    
    return score

def predict_silver_hybrid():
    MODEL_PATH = "silver/models/silver_v1_model.joblib"
    
    # 1. Load latest data
    df = load_silver_data()
    df = generate_silver_features(df)
    last_row = df.iloc[-1]
    
    print(f"🕒 Dữ liệu chốt ngày: {last_row['date'].strftime('%Y-%m-%d')}")
    print(f"💰 GIÁ BẠC PHÚ QUÝ HIỆN TẠI:")
    print(f"   - Mua vào: {last_row['silver_vn_buy']:.2f} triệu VND/lượng")
    print(f"   - Bán ra:  {last_row['silver_vn_sell']:.2f} triệu VND/lượng")
    
    # 2. Expert System
    ex_score = rule_based_silver_expert(last_row)
    ex_dir = 1 if ex_score > 0.05 else (-1 if ex_score < -0.05 else 0)
    
    # 3. ML Model
    ml_dir = 0
    ml_prob = [0.33, 0.33, 0.33]
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        model = bundle['model']
        selector = bundle.get('selector') # V7 có selector
        feat_names = bundle['feature_names']
        
        X = last_row[feat_names].to_frame().T
        
        # Nếu có selector (V7), phải transform dữ liệu trước khi predict
        if selector:
            X_input = selector.transform(X)
        else:
            X_input = X
            
        ml_dir = model.predict(X_input)[0]
        ml_prob = model.predict_proba(X_input)[0]
    
    # 4. Final Decision
    ml_conf = np.max(ml_prob)
    ex_conf = min(abs(ex_score) * 2, 1.0)
    final_score = (ml_dir * ml_conf * 0.65) + (ex_dir * ex_conf * 0.35)
    
    if final_score > 0.15: final_dir = 1
    elif final_score < -0.15: final_dir = -1
    else: final_dir = 0
    
    # 5. Output
    res_map = {1: "TĂNG 📈", -1: "GIẢM 📉", 0: "ĐI NGANG ➡️"}
    conf_score = (ml_conf * 0.65 + ex_conf * 0.35) * 100
    
    print("\n" + " 🏆 DỰ BÁO BẠC PHÚ QUÝ (HYBRID) 🏆 ".center(50, "✨"))
    print(f"🔹 Expert Score: {ex_score:.4f} (Conf: {ex_conf*100:.1f}%)")
    print(f"🔹 AI Model:     {ml_conf*100:.1f}% (Xác suất {res_map[ml_dir]})")
    print("-" * 50)
    print(f"🔮 DỰ BÁO GIÁ BÁN NGÀY MAI: {res_map[final_dir]}")
    print(f"🛡️ ĐỘ TIN CẬY:               {conf_score:.1f}%")
    print("-" * 50)
    
    if final_dir != 0:
        # Bạc biến động ~0.5% - 1% mỗi ngày
        move = final_dir * 0.008 
        pred_sell = last_row['silver_vn_sell'] * (1 + move)
        pred_buy = pred_sell * 0.97 # Duy trì spread 3%
        print(f"🎯 Giá dự kiến ngày mai:")
        print(f"   - Mua vào: ~{pred_buy:.2f} triệu VND/lượng")
        print(f"   - Bán ra:  ~{pred_sell:.2f} triệu VND/lượng")

if __name__ == "__main__":
    predict_silver_hybrid()

