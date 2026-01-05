import os
import csv
import random
import math
from datetime import datetime

# --- CẤU HÌNH GIỚI HẠN NGÀY (Dành cho Backtest) ---
# Nếu muốn dự báo ngày 30/12/2025, hãy đặt là "2025-12-29"
# Nếu muốn dự báo cho ngày mới nhất, hãy đặt là None
BACKTEST_END_DATE = "2025-12-30" 

# --- CÁC ĐỢT CAN THIỆP CỦA NHNN (SBV) ---
SBV_INTERVENTION_DATES = ["2024-05-20", "2024-06-03", "2024-10-15", "2025-01-02"]

def calculate_rsi(prices, n=14):
    if len(prices) < n + 1: return 50
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[-n:]) / n
    avg_loss = sum(losses[-n:]) / n
    if avg_loss == 0: return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def run_gold_expert_system():
    BASE_DIR = os.path.dirname(__file__)
    MASTER_FILE = os.path.join(BASE_DIR, "gold", "data", "gold_master_history.csv")
    
    if not os.path.exists(MASTER_FILE):
        print(f"❌ Không tìm thấy file master.")
        return

    # 1. NẠP DỮ LIỆU & LỌC THEO NGÀY BACKTEST
    raw_data = []
    with open(MASTER_FILE, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if BACKTEST_END_DATE and row['date'] > BACKTEST_END_DATE:
                    continue
                raw_data.append({
                    'date': row['date'],
                    'dxy': float(row['dxy']),
                    'xau_usd': float(row['xau_usd']),
                    'price_w': float(row['xau_vnd_tael']),
                    'price_vn': float(row['sjc_price'])
                })
            except: continue

    if not raw_data:
        print(f"❌ Không có dữ liệu trước ngày {BACKTEST_END_DATE}")
        return

    print(f"🕒 CHẾ ĐỘ BACKTEST: Đang học dữ liệu đến ngày {raw_data[-1]['date']}...")

    # 2. TRÍCH XUẤT ĐẶC TRƯNG
    samples = []
    for i in range(20, len(raw_data)):
        curr, prev = raw_data[i], raw_data[i-1]
        recent_world_prices = [raw_data[j]['xau_usd'] for j in range(i-15, i+1)]
        rsi_w = calculate_rsi(recent_world_prices)
        is_intervention = 1 if curr['date'] in SBV_INTERVENTION_DATES else 0
        past_basis = [(raw_data[j]['price_vn'] - raw_data[j]['price_w']) for j in range(i-20, i)]
        b_mean = sum(past_basis) / 20
        b_std = math.sqrt(sum((x - b_mean)**2 for x in past_basis) / 19)
        basis_z = ((curr['price_vn'] - curr['price_w']) - b_mean) / b_std if b_std > 0 else 0

        def ret(now, p): return (now - p) / p if p else 0
        y_dir = 1 if curr['price_vn'] - prev['price_vn'] > 0.1 else (-1 if curr['price_vn'] - prev['price_vn'] < -0.1 else 0)

        f = {
            'world_rt': ret(curr['xau_usd'], prev['xau_usd']),
            'rsi_w': rsi_w,
            'basis_z': basis_z,
            'is_sbv': is_intervention,
            'vn_mom': ret(prev['price_vn'], raw_data[i-2]['price_vn'])
        }
        samples.append({'x': f, 'y': y_dir, 'date': curr['date']})

    # 3. TRAINING
    train_size = int(len(samples) * 0.85)
    train_samples = samples[:train_size]
    test_samples = samples[train_size:]

    def predict_logic(f, w, t):
        score = (f['world_rt'] * w['world'] + 
                 (w['rsi_up'] if f['rsi_w'] < 35 else (w['rsi_down'] if f['rsi_w'] > 65 else 0)) +
                 f['basis_z'] * w['bz_w'] + (w['sbv_w'] if f['is_sbv'] else 0) + f['vn_mom'] * w['mom'])
        if score > t: return 1
        if score < -t: return -1
        return 0

    best_w, best_t, max_acc = None, 0, 0
    for _ in range(5000):
        w = {'world': random.uniform(100, 300), 'rsi_up': random.uniform(5, 20), 'rsi_down': random.uniform(-20, -5),
             'bz_w': random.uniform(-25, -5), 'sbv_w': random.uniform(-30, -10), 'mom': random.uniform(10, 50)}
        t = random.uniform(0.02, 0.15)
        acc = (sum(1 for s in train_samples if predict_logic(s['x'], w, t) == s['y']) / len(train_samples)) * 100
        if acc > max_acc: max_acc = acc; best_w, best_t = w, t

    # 4. DỰ BÁO NGÀY TIẾP THEO (DỰA TRÊN NGÀY 29/12)
    last_s = samples[-1]
    prediction = predict_logic(last_s['x'], best_w, best_t)
    res = "TĂNG" if prediction == 1 else ("GIẢM" if prediction == -1 else "ĐI NGANG")
    current_p = raw_data[-1]['price_vn']
    
    # Tính giá dự kiến
    avg_change = 0.005
    try:
        changes = [abs((raw_data[j]['price_vn'] - raw_data[j-1]['price_vn'])/raw_data[j-1]['price_vn']) 
                   for j in range(1, len(raw_data)) if abs(raw_data[j]['price_vn'] - raw_data[j-1]['price_vn']) > 0.1]
        if changes: avg_change = sum(changes) / len(changes)
    except: pass
    predicted_p = current_p * (1 + avg_change) if prediction == 1 else (current_p * (1 - avg_change) if prediction == -1 else current_p)

    print("\n" + " 🏆 DỰ BÁO NGÀY 30/12 (DỰA TRÊN DỮ LIỆU ĐẾN 29/12) 🏆 ".center(65, "✨"))
    print(f"🔹 ACCURACY TRONG LỊCH SỬ: {(sum(1 for s in test_samples if predict_logic(s['x'], best_w, best_t) == s['y']) / len(test_samples)) * 100:.2f}%")
    print(f"🔮 DỰ BÁO CHO NGÀY 30/12/2025:")
    print(f"   ▶️ XU HƯỚNG: {res}")
    print(f"   ▶️ GIÁ DỰ KIẾN: {predicted_p:.2f} triệu VND/lượng")
    print(f"   ▶️ GIÁ GỐC (29/12): {current_p:.2f} triệu VND/lượng")
    print("-" * 65)

if __name__ == "__main__":
    run_gold_expert_system()
