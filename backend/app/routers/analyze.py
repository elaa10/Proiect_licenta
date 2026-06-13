from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.models.analysis import AnalysisRequest, AnalysisResult
from app.routers.auth import get_current_user
from app.schemas.analysis import (
    AnalyzeRequest, LexicalResponse, MLResponse,
    ScreenshotResponse,  VisualResponse, FullAnalysisResponse, 
    AnalysisHistoryItem, AnalysisStats, VisualComparisonItem, 
)
from app.services.url_analyzer import extract_features, compute_lexical_score
from app.services.verdict import compute_verdict
from app.services.ml_classifier import is_model_available, predict_ml_score
from app.services.browser_capture import capture_screenshot
from app.services.visual_matcher import is_visual_available, match_brand

from collections import defaultdict

from app.services.visual_matcher_dino import is_dino_available, match_brand_dino

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
    result = match_brand(f"/app/screenshots/{filename}")
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


@router.post("/", response_model=FullAnalysisResponse)
async def analyze_full(
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Full pipeline: lexical + ML + visual. Stores result in DB."""
    _validate_url(payload.url)

    req = AnalysisRequest(user_id=current_user.id, url=payload.url, status="running")
    db.add(req)
    db.commit()
    db.refresh(req)

    lexical_score = None
    ml_score = None
    screenshot_path = None
    visual_brand = None
    visual_similarity = None
    visual_model = payload.visual_model if payload.visual_model in ("clip", "dino") else "clip"

    try:
        # Stage 1 — Lexical
        features = extract_features(payload.url)
        lexical_score = compute_lexical_score(features)

        # Stage 2 — ML
        if is_model_available():
            ml_score = predict_ml_score(payload.url)["score"]

        # Stage 3 — Visual (model chosen by the user)
        filename = await capture_screenshot(payload.url)
        if filename:
            screenshot_path = f"/app/screenshots/{filename}"
            if visual_model == "dino":
                matcher_ready = is_dino_available()
                match = match_brand_dino(screenshot_path) if matcher_ready else None
            else:
                matcher_ready = is_visual_available()
                match = match_brand(screenshot_path) if matcher_ready else None

            if match is not None:
                visual_similarity = match.get("similarity")
                if match["matched"]:
                    visual_brand = match["brand"]

        verdict = compute_verdict(payload.url, lexical_score, ml_score, visual_brand, visual_similarity)["verdict"]

        result = AnalysisResult(
            request_id=req.id,
            lexical_score=lexical_score,
            ml_score=ml_score,
            screenshot_path=screenshot_path,
            visual_match_brand=visual_brand,
            visual_similarity=visual_similarity,
            visual_model=visual_model,
            verdict=verdict,
        )
        db.add(result)
        req.status = "done"
        db.commit()
        db.refresh(result)

        return FullAnalysisResponse(
            request_id=req.id,
            url=req.url,
            lexical_score=lexical_score,
            ml_score=ml_score,
            visual_match_brand=visual_brand,
            visual_similarity=visual_similarity,
            visual_model=visual_model,
            screenshot_path=screenshot_path,
            verdict=verdict,
            created_at=req.created_at,
        )

    except Exception as e:
        req.status = "error"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    

@router.get("/history", response_model=list[AnalysisHistoryItem])
def get_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = 50,
):
    requests = (
        db.query(AnalysisRequest)
        .filter(AnalysisRequest.user_id == current_user.id)
        .order_by(AnalysisRequest.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        AnalysisHistoryItem(
            request_id=r.id,
            url=r.url,
            verdict=r.result.verdict if r.result else None,
            created_at=r.created_at,
        )
        for r in requests
    ]

@router.get("/stats", response_model=AnalysisStats)
def get_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = (
        db.query(AnalysisResult.verdict, func.count(AnalysisResult.id))
        .join(AnalysisRequest, AnalysisResult.request_id == AnalysisRequest.id)
        .filter(AnalysisRequest.user_id == current_user.id)
        .group_by(AnalysisResult.verdict)
        .all()
    )

    counts = {"legitimate": 0, "suspicious": 0, "phishing": 0, "unknown": 0}
    for verdict, count in rows:
        key = verdict if verdict in counts else "unknown"
        counts[key] += count

    return AnalysisStats(total=sum(counts.values()), **counts)

@router.get("/visual/comparisons", response_model=list[VisualComparisonItem])
def get_visual_comparisons(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = (
        db.query(AnalysisRequest.url, AnalysisResult)
        .join(AnalysisResult, AnalysisResult.request_id == AnalysisRequest.id)
        .filter(AnalysisRequest.user_id == current_user.id)
        .filter(AnalysisResult.visual_model.isnot(None))
        .order_by(AnalysisRequest.created_at.desc())
        .all()
    )

    # Keep the most recent result per (url, model)
    latest: dict[tuple[str, str], AnalysisResult] = {}
    for url, result in rows:
        key = (url, result.visual_model)
        if key not in latest:
            latest[key] = result

    by_url: dict[str, dict[str, AnalysisResult]] = defaultdict(dict)
    for (url, model), result in latest.items():
        by_url[url][model] = result

    comparisons = []
    for url, models in by_url.items():
        if "clip" in models and "dino" in models:
            clip_r = models["clip"]
            dino_r = models["dino"]
            comparisons.append(VisualComparisonItem(
                url=url,
                clip_brand=clip_r.visual_match_brand,
                clip_similarity=clip_r.visual_similarity,
                dino_brand=dino_r.visual_match_brand,
                dino_similarity=dino_r.visual_similarity,
                agreement=clip_r.visual_match_brand == dino_r.visual_match_brand,
            ))

    return comparisons