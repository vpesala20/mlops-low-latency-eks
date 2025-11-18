FROM python:3.9-slim-buster

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ADD THE MODEL FILE
COPY model.joblib ./

COPY model.py ./

# Using Gunicorn for stable production entrypoint
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "model:app"]
