import torch

def infer_one_image(model, img_tensor, device, threshold=0.5):
    """
    對單張圖片進行推論 (Inference)
    
    Args:
        model: 訓練好的 U-Net 模型
        image_tensor: 已經 Normalize 過的圖片 (C, H, W)
        device: 'cuda' or 'cpu'
        threshold: 判斷傷口的門檻值 (預設 0.5)
        
    Returns:
        pred_mask: Numpy Array (H, W), 數值只有 0 和 1 (Uint8)
    """
    
    # 1. 切換到評估模式 (關閉 Dropout 等)
    model.eval()
    
    # 2. 停止計算梯度 (省記憶體、加速)
    with torch.no_grad():
        # 3. 增加 Batch 維度
        # 模型訓練時是看一批一批的 (B, C, H, W)
        # 但我們現在只有一張圖 (C, H, W)
        # 所以要用 unsqueeze(0) 把它變成 (1, C, H, W) 假裝成一批
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        # 4. 模型預測
        # 輸出的 logits 範圍是負無限大到正無限大
        logits = model(input_tensor)
        
        # 5. 轉成機率
        # Sigmoid 把 logits 壓縮到 0.0 ~ 1.0 之間
        probs = torch.sigmoid(logits)
        
        # 6. 二值化 (Thresholding)
        # 大於 0.5 變 1.0 (傷口)，小於變 0.0 (背景)
        pred_tensor = (probs > threshold).float()
        
        # 7. 後處理：轉回 Numpy 格式
        # squeeze(): 把剛剛加的 Batch 維度拿掉 (1, 1, H, W) -> (H, W)
        # cpu(): 從顯卡搬回記憶體
        # numpy(): 轉成 numpy array
        # astype('uint8'): 轉成整數 (0 或 1)，節省空間
        pred_mask = pred_tensor.squeeze(0).cpu().numpy().astype("uint8")
    
    return pred_mask