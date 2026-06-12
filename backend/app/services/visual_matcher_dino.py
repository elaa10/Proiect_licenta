import pickle
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

EMBEDDINGS_PATH_DINO = Path("/app/data/brand_embeddings_dino.pkl")

CROP_STRATEGIES = [
    {"name": "top_150",  "top": 0,   "bottom": 150},
    {"name": "top_300",  "top": 0,   "bottom": 300},
    {"name": "top_500",  "top": 0,   "bottom": 500},
    {"name": "mid_300",  "top": 100, "bottom": 400},
]

# Ambiguity guard: a match is accepted only when the top similarity exceeds
# the threshold AND beats the runner-up by at least this margin. Prevents
# misidentification when several brands score very close in the saturated
# region of the embedding similarity space.
MIN_CONFIDENCE_MARGIN = 0.02

_model = None
_processor = None
_embeddings: dict = {}
_lock = threading.Lock()


def _load() -> bool:
    global _model, _processor, _embeddings
    if _model is not None:
        return True
    if not EMBEDDINGS_PATH_DINO.exists():
        return False
    with _lock:
        if _model is not None:
            return True
        try:
            from transformers import AutoImageProcessor, AutoModel
            _processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            _model = AutoModel.from_pretrained("facebook/dinov2-base")
            _model.eval()
            with open(EMBEDDINGS_PATH_DINO, "rb") as f:
                _embeddings = pickle.load(f)
            return True
        except Exception as e:
            print(f"[visual_matcher_dino] load error: {e}")
            return False


def is_dino_available() -> bool:
    return EMBEDDINGS_PATH_DINO.exists()


def _crop_and_embed(img: Image.Image, top: int, bottom: int) -> Optional[np.ndarray]:
    w, h = img.size
    actual_bottom = min(bottom, h)
    actual_top = min(top, actual_bottom)
    if actual_bottom <= actual_top:
        return None
    crop = img.crop((0, actual_top, w, actual_bottom))
    try:
        inputs = _processor(images=crop, return_tensors="pt")
        with torch.no_grad():
            outputs = _model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]  # CLS token
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze().numpy()
    except Exception:
        return None


def _compute_query_embeddings(image_path: str) -> list:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[visual_matcher_dino] image load error: {e}")
        return []

    embeddings = []
    for strategy in CROP_STRATEGIES:
        emb = _crop_and_embed(img, strategy["top"], strategy["bottom"])
        if emb is not None:
            embeddings.append(emb)
    return embeddings


def match_brand_dino(screenshot_path: str, threshold: float = 0.85) -> dict:
    no_match = {"matched": False, "brand": None, "display": None,
                "similarity": 0.0, "label": None}

    if not _load():
        return no_match

    query_embeddings = _compute_query_embeddings(screenshot_path)
    if not query_embeddings:
        return no_match

    # Track best similarity per brand (so the runner-up is from a DIFFERENT brand)
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