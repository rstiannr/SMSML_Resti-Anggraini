FROM python:3.12.7-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["mlflow", "models", "serve", "-m", "model_final/model", "-h", "0.0.0.0", "-p", "5001", "--no-conda"]