from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.routers.auth import get_current_user
from app.schemas.analysis import (
    AnalyzeRequest, LexicalResponse, MLResponse,
    ScreenshotResponse, VisualResponse,
)
from app.services.url_analyzer import extract_features, compute_lexical_score
from app.services.ml_classifier import is_model_available, predict_ml_score
from app.services.browser_capture import capture_screenshot
from app.services.visual_matcher import is_visual_available, match_brand

router = APIRouter(prefix="/analyze", tags=["analyze"])


def _validate_url(url: str) -> None:
    if not url or len(url) > 2048:
        raise HTTPException(status_code=422, detail="Invalid URL.")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=422, detail="URL must start with http:// or https://")


@router.post("/lexical", response_model=LexicalResponse)
def analyze_lexical(
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _validate_url(payload.url)
    features = extract_features(payload.url)
    score = compute_lexical_score(features)
    return LexicalResponse(url=payload.url, score=score, features=features)


@router.post("/ml", response_model=MLResponse)
def analyze_ml(
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _validate_url(payload.url)
    if not is_model_available():
        raise HTTPException(status_code=503, detail="ML model not available. Run scripts/train_rf.py first.")
    result = predict_ml_score(payload.url)
    return MLResponse(url=payload.url, score=result["score"])


@router.post("/screenshot", response_model=ScreenshotResponse)
async def analyze_screenshot(
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _validate_url(payload.url)
    filename = await capture_screenshot(payload.url)
    if filename is None:
        raise HTTPException(status_code=422, detail="Could not capture screenshot.")
    return ScreenshotResponse(
        url=payload.url,
        screenshot=filename,
        screenshot_url=f"/screenshots/{filename}",
    )


@router.post("/visual", response_model=VisualResponse)
async def analyze_visual(
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _validate_url(payload.url)

    if not is_visual_available():
        raise HTTPException(status_code=503, detail="Brand knowledge base not found. Run scripts/init_brand_db.py first.")

    filename = await capture_screenshot(payload.url)
    if filename is None:
        raise HTTPException(status_code=422, detail="Could not capture screenshot.")

    screenshot_path = f"/app/screenshots/{filename}"
    result = match_brand(screenshot_path)

    return VisualResponse(
        url=payload.url,
        screenshot=filename,
        screenshot_url=f"/screenshots/{filename}",
        matched=result["matched"],
        brand=result["brand"],
        display=result["display"],
        similarity=result["similarity"],
        label=result["label"],
    )