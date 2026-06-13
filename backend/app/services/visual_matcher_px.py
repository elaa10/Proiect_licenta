"""
PX CLIP visual matcher — original pixel-based multi-crop strategy.

Goes in: backend/app/services/visual_matcher_px.py

This is a clean re-run of the original pixel-based strategy (top_150, top_300,
top_500, mid_300) with the current clean reference screenshots, to obtain
comparable benchmark numbers. No uniform-color filter is applied so the
comparison with other strategies is isolated to the crop geometry alone.

Loads from /app/data/brand_embeddings_px.pkl — separate from all other pkl files.

Public API:
    is_px_available() -> bool
    match_brand_px(screenshot_path, threshold=0.85) -> dict
"""
import pickle
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import open_clip
import torch
from PIL import Image

EMBEDDINGS_PATH = Path("/app/data/brand_embeddings_px.pkl")

# Original pixel-based crop strategies.
# Coordinates are absolute pixel values (top/bottom from image top edge),
# full image width. Designed for 1280x800 Playwright captures.
CROP_STRATEGIES = [
    {"name": "top_150", "top": 0,   "bottom": 150},
    {"name": "top_300", "top": 0,   "bottom": 300},
    {"name": "top_500", "top": 0,   "bottom": 500},
    {"name": "mid_300", "top": 100, "bottom": 400},
]

MIN_CONFIDENCE_MARGIN = 0.02

_model = None
_preprocess = None
_embeddings: dict = {}
_lock = threading.Lock()


def _load() -> bool:
    global _model, _preprocess, _embeddings
    if _model is not None:
        return True
    if not EMBEDDINGS_PATH.exists():
        return False
    with _lock:
        if _model is not None:
            return True
        try:
            _model, _, _preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            _model.eval()
            with open(EMBEDDINGS_PATH, "rb") as f:
                _embeddings = pickle.load(f)
            return True
        except Exception as e:
            print(f"[visual_matcher_px] load error: {e}")
            return False


def is_px_available() -> bool:
    return EMBEDDINGS_PATH.exists()


def _crop_pixels(
    img: Image.Image,
    top: int,
    bottom: int,
) -> Optional[Image.Image]:
    w, h = img.size
    top    = max(0, min(h, top))
    bottom = max(0, min(h, bottom))
    if bottom <= top:
        return None
    return img.crop((0, top, w, bottom))


def _embed_crop(crop: Image.Image) -> Optional[np.ndarray]:
    try:
        tensor = _preprocess(crop).unsqueeze(0)
        with torch.no_grad():
            emb = _model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze().numpy()
    except Exception:
        return None


def _compute_query_embeddings(image_path: str) -> list:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[visual_matcher_px] image load error: {e}")
        return []

    embeddings = []
    for strategy in CROP_STRATEGIES:
        crop = _crop_pixels(img, top=strategy["top"], bottom=strategy["bottom"])
        if crop is None:
            continue
        emb = _embed_crop(crop)
        if emb is not None:
            embeddings.append(emb)
    return embeddings


def match_brand_px(screenshot_path: str, threshold: float = 0.85) -> dict:
    no_match = {"matched": False, "brand": None, "display": None,
                "similarity": 0.0, "label": None}

    if not _load():
        return no_match

    query_embeddings = _compute_query_embeddings(screenshot_path)
    if not query_embeddings:
        return no_match

    per_brand_best: dict[str, dict] = {}

    for brand_name, brand_data in _embeddings.items():
        brand_best_sim = -1.0
        brand_best_label = None
        for ref in brand_data.get("references", []):
            ref_embeddings = ref.get("embeddings", [])
            if not ref_embeddings:
                single = ref.get("embedding")
                if single is not None:
                    ref_embeddings = [single]

            for ref_emb in ref_embeddings:
                if ref_emb is None:
                    continue
                for query_emb in query_embeddings:
                    sim = float(np.dot(query_emb, ref_emb))
                    if sim > brand_best_sim:
                        brand_best_sim = sim
                        brand_best_label = ref.get("label")

        if brand_best_sim > -1.0:
            per_brand_best[brand_name] = {
                "similarity": brand_best_sim,
                "display": brand_data.get("display", brand_name),
                "label": brand_best_label,
            }

    if not per_brand_best:
        return no_match

    ranked = sorted(per_brand_best.items(), key=lambda x: -x[1]["similarity"])
    top_brand, top_info = ranked[0]
    runner_up_sim = ranked[1][1]["similarity"] if len(ranked) > 1 else 0.0
    margin = top_info["similarity"] - runner_up_sim

    if top_info["similarity"] >= threshold and margin >= MIN_CONFIDENCE_MARGIN:
        return {
            "matched": True,
            "brand": top_brand,
            "display": top_info["display"],
            "similarity": round(top_info["similarity"], 4),
            "label": top_info["label"],
        }

    return {**no_match, "similarity": round(top_info["similarity"], 4)}
