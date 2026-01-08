import requests
import json

def check_api():
    host = "https://price.beaverx.ai"
    url = f"{host}/price-api/vngold/charts/Silver"
    print(f"Checking {url}...")
    try:
        resp = requests.get(url, params={"period": "1y"}, timeout=15)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            # In ra cấu trúc dữ liệu để debug
            if isinstance(data, list):
                print(f"✅ Success! Found {len(data)} data points.")
                print(f"Sample: {data[0]}")
            elif isinstance(data, dict):
                print(f"✅ Success! Keys: {data.keys()}")
                # Thử tìm list dữ liệu
                for k in data.keys():
                    if isinstance(data[k], list):
                        print(f"Found list in key '{k}' with {len(data[k])} items.")
                        break
            return True
    except Exception as e:
        print(f"Error: {e}")
    return False

if __name__ == "__main__":
    check_api()


