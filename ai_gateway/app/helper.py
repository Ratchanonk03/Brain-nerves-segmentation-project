import io
import os
import uuid

import numpy as np
from PIL import Image
import cv2

from fastapi import UploadFile
import albumentations as A

PREPROCESSED_BUCKET = os.getenv("PREPROCESSED_BUCKET")

if PREPROCESSED_BUCKET is None:
    raise RuntimeError("PREPROCESSED_BUCKET not set")

class S3Connector:
    def __init__(self, s3_client):
        self.s3 = s3_client
        
    # ------------------------- Download -----------------------
    def download_from_s3(self, bucket: str, mask_key: str, prob_key: str) -> bytes:
        mask_buffer = io.BytesIO()
        self.s3.download_fileobj(bucket, mask_key, mask_buffer)
        mask_buffer.seek(0)
        
        prob_buffer = io.BytesIO()
        self.s3.download_fileobj(bucket, prob_key, prob_buffer)
        prob_buffer.seek(0)
        
        return mask_buffer, prob_buffer

    def download_multiple_from_s3(self, outputs) -> bytes:
        buffers = []
        for output in outputs:
            bucket = output["bucket"]
            mask_key = output["mask_key"]
            prob_key = output["prob_key"]
            
            print(f"[run_inference] Downloading from S3: bucket={bucket}, maskkey={mask_key}, probkey={prob_key}")
            
    def upload_to_s3(self, bucket: str, image_npy: np.ndarray, key: str, suffix: str = ".npy"):
        if not key.endswith(suffix):
            raise ValueError(f"Key must end with {suffix}")
        
        buffer = io.BytesIO()
        np.save(buffer, image_npy)
        buffer.seek(0)
        
        data = buffer.read()
        self.s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType="application/octet-stream")
        
        return {"bucket": bucket, "key": key}

    # ------------------------- Upload -----------------------

    def upload_multiple_to_s3(self, payloads: list[UploadFile]):
        results = []    
        for payload in payloads:
            image = load_image(payload)
            print(f"[run_inference] Loaded image {payload.filename} size={image.size}")
            
            preprocessed = preprocess_image(image)
            key = uuid.uuid4().hex
            preprocessed_key = f"preprocessed/{key}.npy"
            bucket = PREPROCESSED_BUCKET
            result = self.upload_to_s3(bucket, preprocessed, preprocessed_key)
            results.append(result)
            print(f"[run_inference] Uploading {payload.filename} to S3: bucket={bucket}, key={preprocessed_key}")
            
        return results
        
    

# ---------------------------- Preprocessing ---------------------------
def anonymize_dataset(dataset):
    """Mock anonymization for demo by removing patient name and ID."""
    pass

def dicom_to_image_array(dicom):
    """Mock DICOM to image array conversion for demo."""
    pass

def image_array_to_dicom(image_array):
    """Mock image array to DICOM conversion for demo."""
    pass

def preprocess_image(image_pil: Image.Image, target_size=(256, 256)) -> np.ndarray:
    resize_to_target = A.Resize(
        height=target_size[0], width=target_size[1],
        interpolation=cv2.INTER_LINEAR,
        area_for_downscale="image"
    )
    
    transform = A.Compose([
        resize_to_target,
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])
    
    # dicom_to_image_array() mock for real production
    # anonymize_dataset() mock for real production
    img = np.array(image_pil)
    
    # If grayscale convert to 3-channel
    if img.ndim == 2:  # [H, W]
        img = np.stack([img, img, img], axis=-1)  # [H, W, 3]

    # If RGBA, drop alpha
    if img.shape[-1] == 4: # [H, W, 4]
        img = img[..., :3]

    out = transform(image=img)["image"]  # [H, W, 3]
    out = np.transpose(out, (2, 0, 1))  # [3, H, W]
    
    return out


def load_image(image_file: UploadFile) -> Image.Image:
    if image_file is None:
        raise ValueError("No image provided")

    try:
        image_file.file.seek(0)
        contents = image_file.file.read() 
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except Exception as e:
        raise RuntimeError(f"Error loading image {image_file.filename}: {e}")
    
