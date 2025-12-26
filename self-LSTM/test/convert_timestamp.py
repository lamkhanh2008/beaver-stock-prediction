import datetime

# Timestamp từ API
timestamp_from = 1728604800
timestamp_to = 1731110400

# Chuyển sang datetime
date_from = datetime.datetime.fromtimestamp(timestamp_from)
date_to = datetime.datetime.fromtimestamp(timestamp_to)

print(f"From: {date_from}")  # 2024-10-11 00:00:00
print(f"To: {date_to}")      # 2024-11-09 00:00:00

# Hoặc format đẹp hơn
print(f"From: {date_from.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"To: {date_to.strftime('%Y-%m-%d %H:%M:%S')}")


current_timestamp = int(datetime.datetime.now().timestamp())
now = datetime.datetime.now()

print("=" * 60)
print("TIMESTAMP HIỆN TẠI")
print("=" * 60)
print(f"Thời điểm: {now.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Timestamp: {current_timestamp}")
print()