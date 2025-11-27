import numpy as np
import os
from pathlib import Path
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
import segmentation_models_pytorch as smp
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

THRESHOLD = 0.5
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

class SegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, multiclass=False):
        self.image_paths = list(image_paths)
        self.mask_paths  = list(mask_paths)
        self.transform = transform
        self.multiclass = multiclass

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, i):
        image = str(self.image_paths[i])
        mask = str(self.mask_paths[i])

        image = cv2.imread(image)
        if image is None:
            raise RuntimeError(f"cv2.imread failed for image: {image}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"cv2.imread failed for mask: {mask}")

        if self.transform:
            try:
                out = self.transform(image=image, mask=mask)
            except Exception as e:
                raise RuntimeError(f"Augment failed for\n image: {image}\n mask: {mask}\n error: {e}")
            image, mask = out["image"], out["mask"]

        mask = torch.as_tensor(mask)
        mask = mask.long() if self.multiclass else mask.float()
        mask = (mask / 255.0).unsqueeze(0)  # to [0,1] and add channel dim
        return image, mask
    

    
def precision_score(tp, fp, fn, tn, eps=1e-7):
    return tp / (tp + fp + eps)

def recall_score(tp, fp, fn, tn, eps=1e-7):
    return tp / (tp + fn + eps)

def run_epoch(model,loader, threshold , loss_fn, optimizer, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss, total_iou, total_dice, total_prec, total_rec, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    all_preds, all_targets = [], []

    for imgs, masks in loader:
        imgs, masks = imgs.to(device).float(), masks.to(device).float()
        if train:
            optimizer.zero_grad()

        # Forward pass
        logits = model(imgs)              # [B,1,H,W], raw logits
        loss   = loss_fn(logits, masks)   # scalar

        # ---- Predictions ----
        probs = torch.sigmoid(logits)       # [B,1,H,W]
        preds = (probs > threshold).long().squeeze(1)   # [B,H,W]
        targets = masks.long().squeeze(1)         # [B,H,W]

        # Collect for cm
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

        # ---- Metrics ----
        tp, fp, fn, tn = get_stats(preds, targets, mode="binary")
        iou  = iou_score(tp, fp, fn, tn, reduction="micro").item()
        dice = f1_score(tp, fp, fn, tn, reduction="micro").item()  # Dice == F1
        prec  = precision_score(tp.sum().item(), fp.sum().item(),
                                fn.sum().item(), tn.sum().item())
        rec   = recall_score(tp.sum().item(), fp.sum().item(),
                            fn.sum().item(), tn.sum().item())


        if train:
            loss.backward()
            optimizer.step()

        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        total_iou  += iou * batch_size
        total_dice += dice * batch_size
        total_prec += prec * batch_size
        total_rec  += rec * batch_size
        n += batch_size

    # ---- Final metrics ----
    print("all_preds[0] shape:", all_preds[0].shape)   # expect [B,H,W] or [H,W] if B==1
    print("concat shape:", np.concatenate(all_preds, axis=0).shape)  # expect [N,H,W]
    
    all_preds_con = np.concatenate(all_preds, axis=0)
    all_targets_con = np.concatenate(all_targets, axis=0)

    return {
        "loss":       total_loss / n,
        "iou":        total_iou  / n,
        "dice":       total_dice / n,
        "precision":  total_prec / n,
        "recall":     total_rec  / n,
        "preds":      all_preds_con,
        "targets":    all_targets_con,
    }
    
def download_from_s3(bucket: str, key: str) -> bytes:
    buffer = io.BytesIO()
    s3.download_fileobj(bucket, key, buffer)
    buffer.seek(0)
    return buffer