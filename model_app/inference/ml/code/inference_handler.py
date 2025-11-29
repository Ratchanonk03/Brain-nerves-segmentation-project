import io
import os
import json
import uuid
import numpy as np

import torch
import segmentation_models_pytorch as smp

import boto3
from fastapi import FastAPI

from sagemaker_pytorch_serving_container import default_pytorch_inference_handler
from sagemaker_inference import content_types

RESULTS_BUCKET = os.getenv("RESULTS_BUCKET")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
AWS_REGION = os.getenv("AWS_REGION")

DEFAULT_MODEL_FILENAME = "model.pt"

threshold = float(os.environ.get("THRESHOLD", 0.5))
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=AWS_SESSION_TOKEN,
    region_name=AWS_REGION,
)

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.pt")
    print(f"[model_fn] Loading weights from {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model_fn] Loading model on {device}")

    model = smp.Linknet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    model_path = os.path.join(model_dir, "model.pt")
    print(f"[model_fn] Loading weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

def download_from_s3(bucket: str, key: str) -> bytes:
    print(f"[input_fn] Downloading from S3: bucket={bucket}, key={key}")
    buffer = io.BytesIO()
    s3.download_fileobj(bucket, key, buffer)
    buffer.seek(0)
    return buffer.getvalue()

def upload_to_s3(bucket: str, mask: np.ndarray, mask_key: str, prob: np.ndarray, prob_key: str, suffix: str = ".npy"):
    if not mask_key.endswith(suffix):
        raise ValueError(f"Key must end with {suffix}")
    if not prob_key.endswith(suffix):
        raise ValueError(f"Key must end with {suffix}")
    
    mask_buffer = io.BytesIO()
    np.save(mask_buffer, mask)
    mask_buffer.seek(0)
    
    s3.upload_fileobj(Fileobj=mask_buffer, Bucket=bucket, Key=mask_key, ExtraArgs={"ContentType": "application/octet-stream"})
    
    prob_buffer = io.BytesIO()
    np.save(prob_buffer, prob)
    prob_buffer.seek(0)
    
    s3.upload_fileobj(Fileobj=prob_buffer, Bucket=bucket, Key=prob_key, ExtraArgs={"ContentType": "application/octet-stream"})
    
    return {"bucket": bucket, "mask_key": mask_key, "prob_key": prob_key}

def upload_multiple_to_s3(list_mask: list[np.ndarray], list_prob: list[np.ndarray]):
    if len(list_mask) != len(list_prob):
        raise ValueError("list_mask and list_prob must have same length")

    results = []   
    for mask, prob in zip(list_mask, list_prob):
        mask_key = f"masks/{uuid.uuid4().hex}.npy"
        prob_key = f"probs/{uuid.uuid4().hex}.npy"
        bucket = RESULTS_BUCKET
        
        if bucket is None:
            raise ValueError("RESULTS_BUCKET environment variable is not set")
        
        result = upload_to_s3(bucket, mask, mask_key, prob, prob_key)
        results.append(result)

        print(f"[upload_to_s3] Uploading mask to S3: bucket={bucket}, key={mask_key}")
        print(f"[upload_to_s3] Uploading prob to S3: bucket={bucket}, key={prob_key}")
        
    return results

def input_fn(request_body, content_type):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    data = json.loads(request_body)

    # --- validate required fields ---
    if "inputs" not in data:
        raise ValueError("JSON must contain 'inputs' field")
    
    inputs = []
    for input in data["inputs"]:
        
        if "key" not in input:
            raise ValueError("Each input must contain 'key' field")
        if "bucket" not in input:
            raise ValueError("Each input must contain 'bucket' field")
        
        key = input["key"]
        bucket = input["bucket"]
        
        print(f"[input_fn] Received S3 key: {key} Bucket: {bucket}")
        buffers = download_from_s3(bucket, key)  #bytes buffer
        arr = np.load(io.BytesIO(buffers))
        
        # Expect [C=3, H, W]
        if arr.ndim != 3 or arr.shape[0] != 3:
            raise ValueError(f"Expected preprocessed image of shape [3,H,W], got {arr.shape}")
        
        inputs.append(arr)
    
    # Add batch dim -> [B,3,H,W]
    inputs = np.stack(inputs, 0)
    tensor = torch.from_numpy(inputs)  # float32
    return tensor

@torch.no_grad()
def predict_fn(input_data, model):
    print(f"[predict_fn] Running inference on input data of shape: {input_data.shape}")
    
    device = next(model.parameters()).device
    input_data = input_data.to(device)

    logits = model(input_data)      # output shape [N,1,H,W]
    probs = torch.sigmoid(logits)   # [N,1,H,W] in [0,1]
    print(f"[predict_fn] Inference completed, output probs shape: {probs.shape}")
    return {"probs": probs.cpu().numpy()}

def output_fn(prediction, accept):
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")

    probs = prediction["probs"]  # [N,1,H,W]
    print(f"[output_fn] Processing output probs of shape: {probs.shape}")
    masks = (probs > threshold).astype(np.uint8)  # [N,1,H,W]
    print(f"[output_fn] Generated masks of shape: {masks.shape}")
    N = probs.shape[0]

    # Extract 2D arrays
    list_mask = [masks[i, 0] for i in range(N)]
    list_prob = [probs[i, 0] for i in range(N)]

    # Run async uploader inside sync output_fn
    upload_results = upload_multiple_to_s3(list_mask, list_prob)
    response = {"result": upload_results}
    print(f"[output_fn] output response: {N} results uploaded to S3")

    return json.dumps(response).encode('utf-8'), accept

class LinknetHandler(default_pytorch_inference_handler.DefaultPytorchInferenceHandler):
    VALID_CONTENT_TYPES = (content_types.JSON, content_types.NPY)
    
    @staticmethod
    def _is_model_file(filename):
        is_model_file = False
        if os.path.isfile(filename):
            _, ext = os.path.splitext(filename)
            is_model_file = ext in [".pt", ".pth"]
        return is_model_file
    
    def default_model_fn(self, model_dir):
        return model_fn(model_dir)
    
    def default_input_fn(self, input_data, content_type):
        return input_fn(input_data, content_type)
    
    def default_predict_fn(self, data, model):
        return predict_fn(data, model)
    
    def default_output_fn(self, prediction, accept):
        return output_fn(prediction, accept)
    