FROM python:3.8-slim

WORKDIR /app

# Install system libraries required for LightGBM
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install necessary packages
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy prediction script
COPY predictor.py /app/predictor.py

# Copy machine learning models and other necessary files
COPY best_models_subg.pkl /app/best_models_subg.pkl
COPY xgb_model_grade.pkl /app/xgb_model_grade.pkl
COPY xgbr_int_rate_best.pkl /app/xgbr_int_rate_best.pkl
COPY lgbm_loan_status.pkl /app/lgbm_loan_status.pkl
COPY encoder_subg.json /app/encoder_subg.json
COPY encoder_grade.json /app/encoder_grade.json

CMD ["python", "predictor.py"]
