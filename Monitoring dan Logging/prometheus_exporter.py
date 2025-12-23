import time
import requests
import uvicorn
import psutil
from fastapi import FastAPI, Request
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest
)
from fastapi.responses import Response

app = FastAPI()

HTTP_REQUEST_TOTAL = Counter(
    'http_requests_total', 'Total HTTP Requests'
)
PREDICTION_TOTAL = Counter(
    'model_predictions_total', 'Total Model Predictions'
)
PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds', 'Waktu Prediksi'
)
ERROR_COUNT = Counter(
    'api_errors_total', 'Total API Errors'
)
PRED_LABEL_COUNTS = Counter(
    'model_prediction_labels', 'Hasil Prediksi per Label', ['label']
)

CPU_USAGE = Gauge('system_cpu_usage', 'Persentase CPU')
MEMORY_USAGE = Gauge('system_memory_usage', 'Persentase RAM')


@app.get("/metrics")
async def metrics():
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict")
async def predict(request: Request):
    HTTP_REQUEST_TOTAL.inc()
    start_time = time.time()
    try:
        data = await request.json()
        model_url = "http://127.0.0.1:5001/invocations"

        response = requests.post(model_url, json=data)
        response.raise_for_status()
        result = response.json()

        if 'predictions' in result:
            prediction = result['predictions'][0]
        else:
            prediction = result[0]

        PREDICTION_TOTAL.inc()
        PRED_LABEL_COUNTS.labels(label=str(prediction)).inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)

        return result
    except Exception as e:
        ERROR_COUNT.inc()
        return {"error": str(e)}, 500


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)