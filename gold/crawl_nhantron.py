import os
import requests
import pandas as pd


def parse_world_section(data):
    if not isinstance(data, dict):
        return {}
    wp = data.get("goldPriceWorlds")
    if not isinstance(wp, dict):
        return {}
    return {
        "world_price": wp.get("price"),
        "world_price_vnd": wp.get("goldPriceWorldVND"),
        "world_last_update": wp.get("lastUpdate"),
    }


def parse_histories(data, world_info):
    rows = []
    histories = data.get("goldPriceWorldHistories") if isinstance(data, dict) else None
    if not isinstance(histories, list):
        return pd.DataFrame()
    for item in histories:
        dt = pd.to_datetime(item.get("lastUpdated"), errors="coerce", utc=True)
        if pd.isna(dt):
            continue
        row = {
            "datetime": dt,
            "name": item.get("name"),
            "buy": item.get("buyPrice"),
            "sell": item.get("sellPrice"),
            "zone": item.get("zone"),
        }
        row.update(world_info)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["datetime", "sell"])
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime", "zone"], keep="last")
    return df


def crawl_nhantron(host: str, output_path: str, period: str = "1m", api_key: str = None):
    url = f"{host.rstrip('/')}/price-api/vngold/charts/NhanTron"
    headers = {"accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    params = {"period": period}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("Data") if isinstance(resp.json(), dict) else resp.json()
    world_info = parse_world_section(data if isinstance(data, dict) else {})
    df = parse_histories(data if isinstance(data, dict) else {}, world_info)
    if df.empty:
        print("No data parsed for Nhẫn Trơn.")
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    HOST = os.getenv("GOLD_HOST", "https://price.beaverx.ai")
    API_KEY = os.getenv("GOLD_API_KEY") or os.getenv("API_KEY")
    PERIOD = os.getenv("GOLD_PERIOD", "1m")
    OUTPUT = os.getenv("GOLD_NHANTRON_PATH", "gold/data/nhantron.csv")
    crawl_nhantron(HOST, OUTPUT, period=PERIOD, api_key=API_KEY)
