FROM python:3.9-slim

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ADD THE MODEL FILE
COPY model.joblib ./

COPY model.py ./

# FINAL FIX: Use Gunicorn as the stable production entrypoint
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "model:app"]
