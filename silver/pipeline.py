import os
import pandas as pd
from datetime import datetime
import subprocess
import sys
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# Import config
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config.settings import SILVER_UPDATE_INTERVAL_DAYS
except ImportError:
    SILVER_UPDATE_INTERVAL_DAYS = 2

def run_command(cmd):
    print(f"🚀 Executing: {cmd}")
    workspace_dir = os.getcwd()
    venv_python = os.path.abspath(os.path.join(workspace_dir, "..", "venv", "bin", "python3"))
    
    if not os.path.exists(venv_python):
        venv_python = "python3"
        
    result = subprocess.run([venv_python] + cmd.split(), capture_output=False)
    return result.returncode == 0

def silver_task():
    """Hàm thực thi chính của service"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n🔔 [{timestamp}] SERVICE BẮT ĐẦU KIỂM TRA CHU KỲ DỰ BÁO...")
    
    MASTER_DATA_PATH = "silver/data/silver_master_history.csv"
    need_update = True
    
    if os.path.exists(MASTER_DATA_PATH):
        df = pd.read_csv(MASTER_DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        last_date = df['date'].max()
        days_diff = (datetime.now() - last_date).days
        
        if days_diff < SILVER_UPDATE_INTERVAL_DAYS:
            print(f"🕒 Dữ liệu vẫn còn mới ({days_diff} ngày).")
            need_update = False
        else:
            print(f"🔄 Dữ liệu cũ ({days_diff} ngày). Đang kích hoạt chu trình cập nhật...")
    else:
        print("⚠️ Không tìm thấy dữ liệu. Khởi tạo lần đầu...")

    if need_update:
        # Chu trình Crawl -> Build -> Train
        success = run_command("silver/crawl_data.py") and \
                  run_command("silver/rebuild_master.py") and \
                  run_command("silver/train.py")
        
        if not success:
            print("❌ Chu trình cập nhật thất bại. Vui lòng kiểm tra log.")
            return
            
    # Luôn chạy dự báo
    run_command("silver/predict.py")
    print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] Hoàn thành nhiệm vụ. Đợi chu kỳ tiếp theo...")

if __name__ == "__main__":
    # 1. Chạy ngay lập tức một lần khi khởi động
    silver_task()
    
    # 2. Thiết lập Scheduler để chạy định kỳ (Cronjob trong code)
    scheduler = BlockingScheduler()
    
    # Ví dụ: Chạy vào lúc 08:00 sáng mỗi ngày
    # Bạn có thể điều chỉnh giờ ở đây
    trigger = CronTrigger(hour=8, minute=0) 
    
    scheduler.add_job(silver_task, trigger=trigger, name="SilverDailyUpdate")
    
    print("\n" + "="*60)
    print("🤖 SILVER-SERVICE ĐÃ SẴN SÀNG VÀ ĐANG CHẠY NGẦM...")
    print("📅 Lịch trình: Tự động kiểm tra vào lúc 08:00 mỗi ngày.")
    print("="*60)
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n👋 Đang tắt Service...")
