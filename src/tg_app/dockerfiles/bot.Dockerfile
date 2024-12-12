FROM python:3.11-slim 

RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install -r requirements.txt
RUN pip3 install --force-reinstall -v "aiogram==2.23.1"

COPY service /service

WORKDIR /service
