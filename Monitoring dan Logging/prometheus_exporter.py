import time
import requests
import uvicorn
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

app = FastAPI()

HTTP_REQUEST_TOTAL = Counter('http_requests_total', 'Total HTTP Requests')
PREDICTION_TOTAL = Counter('model_predictions_total',
                           'Total Model Predictions')
PREDICTION_LATENCY = Histogram('model_prediction_latency_seconds',
                               'Time for prediction')
ERROR_COUNT = Counter('api_errors_total', 'Total API Errors')

PRED_COUNTS = {
    0: Counter('pred_label_0_total', 'Total Label 0'),
    1: Counter('pred_label_1_total', 'Total Label 1'),
    2: Counter('pred_label_2_total', 'Total Label 2'),
    3: Counter('pred_label_3_total', 'Total Label 3')
}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict")
async def predict(request: Request):
    HTTP_REQUEST_TOTAL.inc()
    start_time = time.time()

    try:
        data = await request.json()

        model_url = "http://127.0.0.1:5000/invocations"

        response = requests.post(model_url, json=data)
        response.raise_for_status()

        result = response.json()

        prediction_value = (result[0] if isinstance(result, list)
                            else result['predictions'][0])

        PREDICTION_TOTAL.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)

        if prediction_value in PRED_COUNTS:
            PRED_COUNTS[prediction_value].inc()

        return {"prediction": prediction_value}

    except Exception as e:
        ERROR_COUNT.inc()
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
