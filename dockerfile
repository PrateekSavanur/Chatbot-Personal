FROM python:3.10-slim

WORKDIR /app

# Copy dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY app/ .

# Port required by HF Spaces
EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
