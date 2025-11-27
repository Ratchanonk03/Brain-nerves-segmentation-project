import os
import json

from aiohttp import ClientError

import boto3
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

import logging
import app.helper as helper

# --------------------------- Environment Variables ---------------------------
PREPROCESSED_BUCKET = os.getenv("PREPROCESSED_BUCKET")
SAGEMAKER_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
AWS_REGION = os.getenv("AWS_REGION")

app = FastAPI(title="AI Gateway (Segmentation Demo)")
logger = logging.getLogger("uvicorn.error")
# --------------------------- AWS Clients ---------------------------   
s3 = boto3.client("s3", region_name=AWS_REGION, 
                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                  aws_session_token=AWS_SESSION_TOKEN)

sagemaker = boto3.client("sagemaker-runtime",
                        region_name=AWS_REGION, 
                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                        aws_session_token=AWS_SESSION_TOKEN)

s3_connector = helper.S3Connector(s3)

def check_endpoint(name: str):
    if name is None:
        raise RuntimeError("SAGEMAKER_ENDPOINT_NAME not set in .env")
    
    try:
        response = sagemaker.describe_endpoint(EndpointName=name)
        status = response["EndpointStatus"]
        if status != "InService":
            raise RuntimeError(f"SageMaker endpoint '{name}' is not InService (current status: {status})")
    except Exception:
        raise RuntimeError(f"SageMaker endpoint '{name}' does not exist or cannot be accessed")
    
def call_model(inputs: dict, local: bool = False):
    print("[call_model] ENTER with:", inputs)
    if local:
        # local-mode: call your local gateway directly
        import requests
        response = requests.post(
            "http://localhost:8080/invocations",
            json=inputs,
            timeout=30
        )
        return response.json()
    status = None 
    if "inputs" not in inputs:
        raise ValueError("No 'inputs' field provided to call_model()")
    try:
        print("[call_model] Invoking SageMaker endpoint:", SAGEMAKER_ENDPOINT_NAME)
        response = sagemaker.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(inputs)
        )

        status = response["ResponseMetadata"]["HTTPStatusCode"]
        if status != 200:
            raise RuntimeError(f"Bad response from model endpoint")

        return json.loads(response["Body"].read().decode("utf-8"))
    except ClientError as e:
        logger.error(f"[AWS ClientError] {e}")
        raise RuntimeError("Model inference service is unavailable ")
    except json.JSONDecodeError as e:
        logger.error(f"[Model returned invalid JSON] {e}")
        raise RuntimeError("Model output is invalid")
    except Exception as e:
        logger.error(f"[Unexpected error] {e}")
        raise RuntimeError("Internal model invocation error")

def check_s3_health(bucket_name: str) -> bool:
    try:
        s3.head_bucket(Bucket=bucket_name)
        return True
    except Exception as e:
        print("[S3 Health Error]", e)  # log internally
        return False
    
def check_model_health() -> bool:
    try:
        resp = sagemaker.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT_NAME,
            ContentType="application/json",
            Body='{"ping": true}' 
        )
        code = resp["ResponseMetadata"]["HTTPStatusCode"]
        print("Model health HTTP code:", code)
        return code == 200

    except Exception as e:
        print("[Model Health Error]", repr(e))
        return False


# --------------------------- FastAPI endpoints ---------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/model_health")
def model_health_check():
    try:
        response = sagemaker.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps({"ping": True}),
        )

        code = response["ResponseMetadata"]["HTTPStatusCode"]

        if code == 200:
            return {"status": "ok", "model": "healthy"}
        else:
            return {"status": "degraded", "model": "unhealthy"}

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "error": str(e)}
        )

@app.get("/s3_health")
def s3_health_check():
    s3_preprocess = check_s3_health(PREPROCESSED_BUCKET)
    
    health = {
        "s3_preprocess": "healthy" if s3_preprocess else "unhealthy"
        }

    if not s3_preprocess:
        raise HTTPException(status_code=503, detail={"status": "degraded", **health})

    return {"status": "ok", **health}
    
@app.post("/infer")
def run_inference(payloads: list[UploadFile] = File(...)):
    try:      
        model_payloads = {"inputs": s3_connector.upload_multiple_to_s3(payloads)}
        
        model_response = call_model(model_payloads, True)
        
        return JSONResponse(content=model_response)
    
    except RuntimeError as e:
        # expected / handled errors (bad model response, bad image, etc.)
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        # unexpected system errors
        raise HTTPException(status_code=500, detail=f"Server error: {e}")