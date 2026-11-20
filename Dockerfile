# Runtime-only image for bench.py (stand-in for a Raspberry Pi 4). Versions pinned to match the models' training env.
FROM python:3.11-slim
RUN pip install --no-cache-dir numpy==2.4.6 scikit-learn==1.9.1 xgboost==3.2.0 psutil joblib tensorflow-cpu==2.21.0 keras==3.15.1
WORKDIR /app
COPY bench.py ./
COPY src/__init__.py src/common.py src/predictors.py ./src/
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3
