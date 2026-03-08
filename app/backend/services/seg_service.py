import time
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ..core import DEVICE, IMG_SIZE, NORM_MEAN, NORM_STD
from ..models import SegmentationModelManager


class SegmentationService:
    
    @staticmethod
    def get_pred_transforms():
        return A.Compose([
            A.Resize(height=IMG_SIZE[1], width=IMG_SIZE[0]),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ToTensorV2()
        ])
    
    @classmethod
    async def predict_mask(cls, image_bytes, version):
        start_time = time.time()
        
        # 1. 將前端傳來的 bytes 轉為 OpenCV RGB 格式 (對齊你的 Dataset)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("無法解析影像檔案")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. 透過 Manager 取得指定版本的模型 (含快取機制)
        model = SegmentationModelManager.get_model(version)
        
        # 3. 套用 Albumentations 預處理
        transform = cls.get_pred_transforms()
        augmented = transform(image=img)
        input_tensor = augmented["image"].unsqueeze(0).to(DEVICE)
        
        # 4. 執行推論 (不計算梯度，節省 A10 顯存)
        with torch.no_grad():
            # 模型輸出：原始分數
            logits = model(input_tensor)
            
            # 轉成機率：把模型輸出的原始分數（logits）由 sigmoid 0.0 ~ 1.0 的機率
            probs = torch.sigmoid(logits)
            
            # 二值化：把 > 0.5 變成 1.0, < 0.5 變成 0.0
            mask = (probs > 0.5).float()
            
            # 最終結果遮罩：把 batch 與 channel 維度拿掉並搬回 cpu 最後把它轉成 numpy 型態為 unsigned int 8 bit
            mask = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
        
        wound_area_px = int(np.sum(mask == 255))
        total_px = mask.size
        wound_ratio = round(wound_area_px / total_px * 100, 2)
        inference_time = round(time.time() - start_time, 4)
        
        metrics = {
            "version": version,
            "inference_time": inference_time,
            "wound_area_px": wound_area_px,
            "wound_ratio": wound_ratio
        }
        
        return mask, metrics
    
    
    @staticmethod
    def get_overlay(image_bytes, mask: np.ndarray, alpha: float):
        nparr = np.frombuffer(image_bytes, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        if original_img is None:
            raise ValueError("無法解析影像檔案")
        
        h, w = original_img.shape[:2]
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        mask_color = np.zeros_like(original_img)
        mask_color[mask_resized == 255] = (0, 255, 0) # Make predicted part green.
        overlay = cv2.addWeighted(original_img, 1, mask_color, alpha, 0)
        
        contours, _ = cv2.findContours(
            mask_resized,
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
        
        overlay = overlay.astype(np.uint8)
        
        return np.ascontiguousarray(overlay, dtype=np.uint8)