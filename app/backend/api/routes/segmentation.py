import io
import cv2
import base64
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from ...services.seg_service import SegmentationService


router = APIRouter()


@router.post("/predict")
async def predict_mask(
    file: UploadFile = File(...),
    version: str = Form("v3")
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="❌ 檔案格式錯誤，請上傳圖片")
    
    try:
        image_bytes = await file.read()
        pred_mask, metrics = await SegmentationService.predict_mask(image_bytes, version)
        
        _, buffer = cv2.imencode(".png", pred_mask)
        io_buf = io.BytesIO(buffer)
        
        return StreamingResponse(io_buf, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"🔥 推論失敗: {str(e)}")


@router.post("/overlay")
async def predict_overlay(
    file: UploadFile = File(...),
    version: str = Form("v3"),
    alpha: float = Form(0.15)
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="❌ 檔案格式錯誤，請上傳圖片")
    
    try:
        image_bytes = await file.read()
        pred_mask, metrics = await SegmentationService.predict_mask(image_bytes, version)
        overlay_img = SegmentationService.get_overlay(image_bytes, pred_mask, alpha)
        
        success, buffer = cv2.imencode(".png", overlay_img)
        if not success:
            raise ValueError("圖性編碼失敗")
        
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "data": {
                "image": f"data:image/png;base64,{img_base64}", # 前端 <img> 標籤直接可用
                "analysis": metrics
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"🔥 推論失敗: {str(e)}")