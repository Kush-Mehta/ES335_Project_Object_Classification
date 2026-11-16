FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
COPY vision_features.py ./
COPY results/ ./results/
ENV MODEL_DIR=/app/results
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
