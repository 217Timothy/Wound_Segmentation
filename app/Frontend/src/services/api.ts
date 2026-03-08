interface AnalysisData {
  wound_area_px: number;
  wound_ratio: number;
}

interface PredictionResponse {
  status: string;
  data: {
    image: string;
    analysis: AnalysisData;
  };
}

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
