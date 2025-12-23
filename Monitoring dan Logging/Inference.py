import requests
import time
import random

url = 'http://localhost:8000/predict'
headers = {'Content-Type': 'application/json'}

data_samples = [
    [5.5, 1.2, 10.0, 2.5],
    [500.0, 150.0, 1500.0, 75.0],
    [2500.0, 800.0, 5000.0, 200.0]
]

columns = ["Avg_Sales", "Std_Dev", "Max_Sales", "UnitPrice"]

while True:
    try:
        selected_data = random.choice(data_samples)
        payload = {
            "dataframe_split": {
                "columns": columns,
                "data": [selected_data]
            }
        }
        
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            print(f"✅ Sukses! Input: {selected_data} "
                  f"-> Prediksi: {response.json()}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"⚠️ Gagal terhubung: {e}")
        time.sleep(2)
        
    time.sleep(1)