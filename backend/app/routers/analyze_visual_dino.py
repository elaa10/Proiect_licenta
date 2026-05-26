from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.routers.auth import get_current_user
from app.schemas.analysis import AnalyzeRequest, VisualResponse
from app.services.browser_capture import capture_screenshot
from app.services.visual_matcher_dino import is_dino_available, match_brand_dino

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("/visual/dino", response_model=VisualResponse)
async def analyze_visual_dino(
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not is_dino_available():
        raise HTTPException(
            status_code=503,
            detail="DINOv2 knowledge base not found. Run scripts/init_brand_db_dino.py first."
        )

    filename = await capture_screenshot(payload.url)
    if filename is None:
        raise HTTPException(status_code=422, detail="Could not capture screenshot.")

    result = match_brand_dino(f"/app/screenshots/{filename}")

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