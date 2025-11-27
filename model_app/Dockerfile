# syntax=docker/dockerfile:1.6
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime AS base-amd64

WORKDIR /opt/ml/code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference/inference.py inference/inference.py

RUN mkdir -p /opt/ml/model
COPY inference/model.pt /opt/ml/model/model.pt

ENV SAGEMAKER_PROGRAM=inference/inference.py
ENV PYTHONUNBUFFERED=1

CMD ["python", "inference/inference.py"]
