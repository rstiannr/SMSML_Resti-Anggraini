import requests
import time
import random

url = 'http://localhost:8000/predict'
headers = {'Content-Type': 'application/json'}

data_samples = [
    [100.0, 5.0, 100, 0.5, 5.0],
    [10.0, 1.0, 500, 0.8, 10.0],
    [200.0, 20.0, 600, 1.0, 10.0],
    [100.0, 25.0, 100, 2.5, 5.0],
    [10.0, 3.0, 600, 3.0, 10.0],
    [10.0, 0.5, 50, 0.5, 5.0],
    [5.0, 0.1, 10, 0.2, 2.0],
    [20.0, 2.0, 100, 1.0, 3.0],
    [10.0, 2.0, 50, 2.0, 5.0],
    [20.0, 7.0, 100, 3.5, 3.0]
]

columns = ["Avg_Sales", "Std_Dev", "Max_Sales", "CV", "UnitPrice"]

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