FROM python:3.11-slim 

RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install --no-cache-dir gdown==4.6.0

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY service /service

RUN gdown --folder 1-1Kpyal56J86sZ0dOHZpg_gP2HlkAPys -O /service/tokenizer
RUN gdown 1-T2qx6nZHf0PtqwStPUinhGpC2xSDxQY -O /service/tokenizer/id2label2.json

WORKDIR /service
