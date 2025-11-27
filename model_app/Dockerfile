FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends\
    python3-pip \
    python3-dev \
    build-essential \
    python3-opencv \
    libglib2.0-0 \
    openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/ml/model/code

COPY requirements.txt /opt/ml/model/code/requirements.txt

RUN pip install --no-cache-dir -r /opt/ml/model/code/requirements.txt 

WORKDIR /model_app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1


COPY inference/code/default_pytorch_inference_handler.py /opt/ml/model/code/default_pytorch_inference_handler.py
COPY inference/code/handler_service.py /opt/ml/model/code/handler_service.py
COPY inference/code/server.py /opt/ml/model/code/server.py

EXPOSE 8080

ENTRYPOINT ["python", "-m", "sagemaker_inference"]

CMD ["serve"]