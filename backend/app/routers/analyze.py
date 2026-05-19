from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.analysis import AnalysisRequest, AnalysisResult
from app.models.user import User
from app.routers.auth import get_current_user
from app.schemas.analysis import AnalyzeRequest, LexicalResponse
from app.services.url_analyzer import extract_features, compute_lexical_score

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("/lexical", response_model=LexicalResponse)
def analyze_lexical(
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not payload.url or len(payload.url) > 2048:
        raise HTTPException(status_code=422, detail="Invalid URL")

    features = extract_features(payload.url)
    score = compute_lexical_score(features)

    return LexicalResponse(url=payload.url, score=score, features=features)