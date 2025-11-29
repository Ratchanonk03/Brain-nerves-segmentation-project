FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends\
    python3-pip \
    python3-dev \
    build-essential \
    python3-opencv \
    libglib2.0-0 \
    openjdk-11-jre-headless \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/ml/model && mkdir -p /opt/ml/code

COPY requirements.txt /opt/ml/code/requirements.txt

RUN pip install --no-cache-dir --timeout 300 --retries 5 -r /opt/ml/code/requirements.txt 

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:${PATH}"

COPY inference/code/inference_handler.py /opt/ml/code/inference_handler.py
COPY inference/code/handler_service.py /opt/ml/code/handler_service.py
COPY inference/code/server.py /opt/ml/code/server.py

EXPOSE 8080

WORKDIR /opt/ml/code

ENTRYPOINT ["python3", "server.py"]

CMD ["serve"]