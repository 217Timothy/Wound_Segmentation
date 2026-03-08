import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[2]
# sys.path.append(str(ROOT)) 
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .api.routes import segmentation


app = FastAPI(
    title="Wound Segmentation API",
    description="Wound Segmentation Service based on our model (3 versions: v1/ v2/ v3)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 生產環境建議設為特定網址
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(segmentation.router, prefix="/api/v1", tags=["Segmentation"])

@app.get("/")
def health_check():
    data = {
        "status": "online", 
        "message": "傷口分割伺服器運作中",
        "supported_versions": ["v1", "v2", "v3"]
    }
    
    return JSONResponse(content=data)