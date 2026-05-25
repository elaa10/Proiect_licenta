from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.routers.auth import get_current_user
from app.schemas.analysis import AnalyzeRequest, LexicalResponse, MLResponse
from app.services.url_analyzer import extract_features, compute_lexical_score
from app.services.ml_classifier import is_model_available, predict_ml_score

router = APIRouter(prefix="/analyze", tags=["analyze"])


def _validate_url(url: str) -> None:
    if not url or len(url) > 2048:
        raise HTTPException(status_code=422, detail="Invalid URL")


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
        raise HTTPException(
            status_code=503,
            detail="ML model not available. Run scripts/train_rf.py to train it.",
        )
    result = predict_ml_score(payload.url)
    return MLResponse(url=payload.url, score=result["score"])