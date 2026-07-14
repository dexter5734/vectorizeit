FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcairo2 \
        libcairo2-dev \
        libffi-dev \
        libglib2.0-0 \
        libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
RUN mkdir -p uploads outputs

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
