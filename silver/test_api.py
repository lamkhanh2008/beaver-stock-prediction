import requests
import os

def test_silver_api():
    host = "https://price.beaverx.ai"
    # Thử nghiệm một số endpoint tiềm năng cho bạc
    endpoints = ["Silver", "XAG", "Bạc", "PNJ_Silver"]
    
    for ep in endpoints:
        url = f"{host}/price-api/vngold/charts/{ep}"
        print(f"Testing {url}...")
        try:
            resp = requests.get(url, params={"period": "1y"}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data and (isinstance(data, list) or (isinstance(data, dict) and "Data" in data)):
                    print(f"✅ Found data for: {ep}")
                    return ep
            else:
                print(f"❌ Failed: {resp.status_code}")
        except Exception as e:
            print(f"⚠️ Error: {e}")
    return None

if __name__ == "__main__":
    found = test_silver_api()
    if not found:
        print("Could not find silver data on API.")


