import cv2
import numpy as np
import torch


def tensor_to_numpy(tensor):
    """
    將 PyTorch Tensor (C, H, W) 轉換為 OpenCV 圖片格式 (H, W, C)
    
    Args:
        tensor: 模型輸入的 Tensor，數值範圍通常是 0.0 ~ 1.0 (Float)
    Returns:
        img: Numpy array, 數值範圍 0 ~ 255 (Uint8), 格式為 (H, W, C)
    """
    
    img = tensor.permute(1, 2, 0).cpu().numpy() # (C, H, W) -> (H, W, C)
    
    img = (img * 225).clip(0, 225).astype(np.uint8) # clip 是為了防止有些數值稍微小於 0 或大於 1
    
    return img


def make_overlay(img, mask, color=(0, 255, 0), alpha=0.5):
    """
    將 Mask 疊加在圖片上
    color: 預設為綠色 (0, 255, 0) 因為傷口通常是紅的，綠色對比最明顯
    """
    
    mask_color = np.zeros_like(img)
    mask_color[mask == 255] = color
    
    img_cpy = img.copy()
    overlay = cv2.addWeighted(img_cpy, 1, mask_color, alpha, 0)
    
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(
        overlay,
        contours,
        -1,
        color=(0, 255, 255),
        thickness=2
    )
    
    return overlay


def make_combine(img, pred, gt=None):
    """
    拼接圖片，用來做對比：[原圖 | 真實標籤 | 預測結果]
    
    Args:
        image: 原圖 (H, W, 3)
        pred: 預測遮罩 (H, W)
        gt: 真實標籤 (H, W), 選填
    Returns:
        combined: 拼接後的大圖 [original image | predict mask] 或是 [Original Image | Predict Mask | Ground Truth]
    """
    
    pred_vis = cv2.cvtColor(pred * 255, cv2.COLOR_GRAY2BGR)
    
    if gt:
        gt_vis = cv2.cvtColor(gt * 255, cv2.COLOR_GRAY2BGR)
        combine = np.hstack((img, pred_vis, gt_vis))
    else:
        combine = np.hstack((img, pred_vis))
    
    return combine