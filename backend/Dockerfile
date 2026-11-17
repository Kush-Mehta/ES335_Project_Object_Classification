FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
COPY vision_features.py ./
RUN mkdir -p ./results
COPY results/best_params_twostream.json ./results/
COPY results/model_logisticregression_hog.joblib results/model_logisticregression_color.joblib ./results/
COPY results/model_svm_rbf_hog.joblib results/model_svm_rbf_color.joblib ./results/
COPY results/model_randomforest_hog.joblib results/model_randomforest_color.joblib ./results/
COPY results/scaler_hog.joblib results/scaler_color.joblib results/label_encoder.joblib ./results/
ENV MODEL_DIR=/app/results
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
