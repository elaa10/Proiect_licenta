from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class AnalyzeRequest(BaseModel):
    url: str


class LexicalResponse(BaseModel):
    url: str
    score: float
    features: Dict[str, Any]


class MLResponse(BaseModel):
    url: str
    score: float


class ScreenshotResponse(BaseModel):
    url: str
    screenshot: str
    screenshot_url: str


class VisualResponse(BaseModel):
    url: str
    screenshot: str
    screenshot_url: str
    matched: bool
    brand: Optional[str]
    display: Optional[str]
    similarity: float
    label: Optional[str]


class FullAnalysisResponse(BaseModel):
    request_id: int
    url: str
    lexical_score: Optional[float]
    ml_score: Optional[float]
    visual_match_brand: Optional[str]
    visual_similarity: Optional[float]
    screenshot_path: Optional[str]
    verdict: str
    created_at: datetime

    class Config:
        from_attributes = True


class AnalysisHistoryItem(BaseModel):
    request_id: int
    url: str
    verdict: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True