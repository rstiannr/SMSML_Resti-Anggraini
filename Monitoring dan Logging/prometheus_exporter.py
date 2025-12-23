import time
import requests
import uvicorn
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
ERROR_COUNT = Counter(
    'api_errors_total', 'Total API Errors'
)
PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds', 'Prediction latency'
)
PREDICTION_TOTAL = Counter(
    'model_predictions_total', 'Total Model Predictions'
)
PRED_LABEL_COUNTS = Counter(
    'model_prediction_labels', 'Labels count', ['label']
)
CURRENT_PREDICTION_VALUE = Gauge(
    'current_prediction_label', 'Last prediction label'
)


@app.get("/metrics")
async def metrics():
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

        if isinstance(result, dict) and 'predictions' in result:
            prediction = result['predictions'][0]
        elif isinstance(result, list):
            prediction = result[0]
        else:
            prediction = result

        PREDICTION_TOTAL.inc()
        PRED_LABEL_COUNTS.labels(label=str(prediction)).inc()
        CURRENT_PREDICTION_VALUE.set(prediction)
        PREDICTION_LATENCY.observe(time.time() - start_time)

        return result

    except Exception as e:
        ERROR_COUNT.inc()
        return {"error": str(e)}, 500


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)