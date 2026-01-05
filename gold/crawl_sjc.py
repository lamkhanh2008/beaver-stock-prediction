import os
import argparse
import requests
import pandas as pd
from datetime import datetime, timezone


def parse_sjc_response(data):
    rows = []
    # Chấp nhận nhiều format: {"data": [...]}, {"Data": {...}}, {"result": [...]}, hoặc lồng thêm 1 dict
    if isinstance(data, dict):
        if "data" in data:
            data = data["data"]
        elif "Data" in data:
            data = data["Data"]
        elif "result" in data:
            data = data["result"]

        # Nếu vẫn là dict, thử lấy list từ các key phổ biến
        if isinstance(data, dict):
            for key in [
                "data",
                "list",
                "items",
                "results",
                "goldPriceSjcHistories",
                "goldPriceSjcHistory",
                "goldPriceWorldHistories",
            ]:
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break

    if not isinstance(data, list):
        return pd.DataFrame()
    for item in data:
        try:
            dt = pd.to_datetime(item.get("createdAt"), utc=True)
        except Exception:
            continue
        rows.append(
            {
                "datetime": dt,
                "name": item.get("name"),
                "buy": item.get("buyPrice"),
                "sell": item.get("sellPrice"),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).dropna(subset=["datetime", "sell"])
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    return df


def crawl_sjc(host: str, output_path: str, period: str = "1m", api_key: str = None, start_date: str = None, end_date: str = None):
    url = f"{host.rstrip('/')}/price-api/vngold/charts/SJC"
    headers = {"accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    
    # Nếu có start_date và end_date, ta sẽ chuyển sang lấy period lớn hơn để đảm bảo có dữ liệu rồi lọc
    params = {"period": period}
    if start_date or end_date:
        params["period"] = "1y" # Lấy 1 năm để bao phủ khoảng thời gian cần lọc

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = parse_sjc_response(data)
    
    if df.empty:
        print("No data parsed for SJC.")
        return

    # Thực hiện lọc theo ngày nếu có yêu cầu
    if start_date:
        df = df[df['datetime'] >= pd.to_datetime(start_date).tz_localize('UTC')]
    if end_date:
        df = df[df['datetime'] <= pd.to_datetime(end_date).tz_localize('UTC')]

    if df.empty:
        print(f"⚠️ Không có dữ liệu SJC trong khoảng {start_date} -> {end_date}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Xử lý Append dữ liệu: Giữ nguyên file cũ, thêm dữ liệu mới, xóa trùng và sắp xếp
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        existing_df['datetime'] = pd.to_datetime(existing_df['datetime'], utc=True)
        # Gộp dữ liệu
        final_df = pd.concat([existing_df, df])
    else:
        final_df = df

    # Sắp xếp theo thời gian
    final_df = final_df.sort_values('datetime')
    
    # CHỈ GIỮ LẠI GIÁ CUỐI CÙNG CỦA MỖI NGÀY (Lọc theo cột Date thay vì DateTime)
    final_df['date_only'] = final_df['datetime'].dt.date
    final_df = final_df.drop_duplicates(subset=['date_only'], keep='last').drop(columns=['date_only'])
    
    # Lưu lại file
    final_df.to_csv(output_path, index=False)
    print(f"✅ Đã cập nhật file {output_path}. Tổng cộng: {len(final_df)} dòng (Dữ liệu mới: {len(df)} dòng).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("GOLD_HOST", "https://price.beaverx.ai"))
    parser.add_argument("--output", default=os.getenv("GOLD_SJC_PATH", "gold/gold/data/sjc.csv"))
    parser.add_argument("--period", default=os.getenv("GOLD_PERIOD", "1m"), help="e.g. 1m, 3m, 6m, 1y")
    parser.add_argument("--api-key", default=os.getenv("GOLD_API_KEY") or os.getenv("API_KEY"))
    parser.add_argument("--start", help="Ngày bắt đầu (YYYY-MM-DD)")
    parser.add_argument("--end", help="Ngày kết thúc (YYYY-MM-DD)")
    args = parser.parse_args()
    
    crawl_sjc(args.host, args.output, period=args.period, api_key=args.api_key, start_date=args.start, end_date=args.end)
