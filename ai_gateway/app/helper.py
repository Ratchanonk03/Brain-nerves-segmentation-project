import io
import os
import uuid

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from fastapi import UploadFile
import albumentations as A

class S3Connector:
    def __init__(self, s3_client, preprocessed_bucket):
        self.s3 = s3_client
        self.preprocessed_bucket = preprocessed_bucket
        
    # ------------------------- Download -----------------------
    def download_from_s3(self, bucket: str, mask_key: str, prob_key: str):
        mask_buffer = io.BytesIO()
        self.s3.download_fileobj(bucket, mask_key, mask_buffer)
        mask_buffer.seek(0)
        
        prob_buffer = io.BytesIO()
        self.s3.download_fileobj(bucket, prob_key, prob_buffer)
        prob_buffer.seek(0)
        
        return mask_buffer, prob_buffer

    def download_multiple_from_s3(self, outputs):
        mask_buffers = []
        prob_buffers = []
        # outputs has format: {"result": [{"bucket": ..., "mask_key": ..., "prob_key": ...}]}
        results = outputs.get("result", outputs) if isinstance(outputs, dict) else outputs
        for result in results:
            bucket = result["bucket"]
            mask_key = result["mask_key"]
            prob_key = result["prob_key"]
            
            mask_buffer, prob_buffer = self.download_from_s3(bucket, mask_key, prob_key)
            mask_buffers.append(mask_buffer)
            prob_buffers.append(prob_buffer)
            
            print(f"[run_inference] Downloading from S3: bucket={bucket}, maskkey={mask_key}, probkey={prob_key}")
            
        return mask_buffers, prob_buffers
    # ------------------------- Upload -----------------------
    
    def upload_to_s3(self, bucket: str, image_npy: np.ndarray, key: str, suffix: str = ".npy"):
        if not key.endswith(suffix):
            raise ValueError(f"Key must end with {suffix}")
        
        buffer = io.BytesIO()
        np.save(buffer, image_npy)
        buffer.seek(0)
        
        data = buffer.read()
        self.s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType="application/octet-stream")
        
        return {"bucket": bucket, "key": key}

    def upload_multiple_to_s3(self, payloads: list[UploadFile]):
        results = []    
        original_sizes = []  # Store original sizes
        for payload in payloads:
            image = load_image(payload)
            original_sizes.append(image.size)  # (width, height)
            print(f"[run_inference] Loaded image {payload.filename} size={image.size}")
            
            preprocessed = preprocess_image(image)
            key = uuid.uuid4().hex
            preprocessed_key = f"preprocessed/{key}.npy"
            bucket = self.preprocessed_bucket
            result = self.upload_to_s3(bucket, preprocessed, preprocessed_key)
            results.append(result)
            print(f"[run_inference] Uploading {payload.filename} to S3: bucket={bucket}, key={preprocessed_key}")
            
        return results, original_sizes

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
    resize_to_256 = A.Resize(
        height=256, width=256,
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        area_for_downscale="image_mask"
    )
        
    transform = A.Compose([
        resize_to_256,
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
    print(f"[preprocess_image] After padding shape: {out.shape}")
    
    # Ensure float32 dtype for model compatibility
    out = out.astype(np.float32)
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
   
def deprocess_image(image_tensor: np.ndarray) -> np.ndarray:
    """Convert model output tensor to displayable image array [H,W,3] uint8."""
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError(f"Expected image tensor of shape [3,H,W], got {image_tensor.shape}")
    
    # [3,H,W] -> [H,W,3]
    image = np.transpose(image_tensor, (1, 2, 0))
    
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image * std) + mean  # reverse normalization
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    
    return image

def overlay(image_rgb: np.ndarray, mask_bin: np.ndarray, alpha: float = 0.5, color_array=[255, 0, 0]):
    # Ensure mask is 2D and binary
    mask = mask_bin.squeeze()
    mask = (mask > 0.5).astype(np.uint8)

    # Convert image to uint8 if needed
    image = image_rgb.copy()
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

    # Create overlay image
    overlay = image.copy()
    overlay[mask == 1] = color_array

    # Alpha blending
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    return blended

def convert(mask_buffer: io.BytesIO, prob_buffer: io.BytesIO):
    mask_array = np.load(mask_buffer)  # [H,W] uint8
    prob_array = np.load(prob_buffer)  # [H,W] float32

    if mask_array.ndim != 2:
        raise ValueError(f"Expected mask of shape [H,W], got {mask_array.shape}")
    if prob_array.ndim != 2:
        raise ValueError(f"Expected prob of shape [H,W], got {prob_array.shape}")
    return mask_array, prob_array

def save_result(mask_arr, prob_arr, save_path: str, identifier: str = "", original_size=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Resize back to original image size if provided
    if original_size is not None:
        width, height = original_size
        mask_arr = cv2.resize(mask_arr, (width, height), interpolation=cv2.INTER_NEAREST)
        prob_arr = cv2.resize(prob_arr, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Save original mask (threshold=0.5) as proper grayscale PNG
    mask_img = (mask_arr * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_img, mode='L')
    mask_pil.save(os.path.join(save_path, f"mask_{identifier}.png"))
    
    # Save probability map with color mapping
    plt.imsave(os.path.join(save_path, f"probability_{identifier}.png"), prob_arr, cmap='inferno')
    
    