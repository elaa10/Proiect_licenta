from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


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
    visual_model: Optional[str] = None

    class Config:
        from_attributes = True


class AnalysisHistoryItem(BaseModel):
    request_id: int
    url: str
    verdict: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class AnalysisStats(BaseModel):
    total: int
    legitimate: int
    suspicious: int
    phishing: int
    unknown: int

class AnalyzeRequest(BaseModel):
    url: str
    visual_model: Optional[str] = "clip"

class VisualComparisonItem(BaseModel):
    url: str
    clip_brand: Optional[str]
    clip_similarity: Optional[float]
    dino_brand: Optional[str]
    dino_similarity: Optional[float]
    agreement: bool