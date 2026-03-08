from pydantic import BaseModel
from typing import List, Optional


class SegmentationResponse(BaseModel):
    status: str = "success"
    version: str               # 模型版本 (v1/v2/v3)
    inference_time: float      # 推論耗時 (秒)
    wound_area_px: int         # 傷口佔據的像素量
    wound_ratio: float         # 傷口占整張圖的百分比 (%)
    message: Optional[str] = None