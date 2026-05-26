import pickle
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import open_clip
import torch
from PIL import Image

EMBEDDINGS_PATH = Path("/app/data/brand_embeddings.pkl")

# Multiple crop strategies — each captures a different visual region
CROP_STRATEGIES = [
    {"name": "top_150",  "top": 0,    "bottom": 150},   # logo/navbar only
    {"name": "top_300",  "top": 0,    "bottom": 300},   # header area (baseline)
    {"name": "top_500",  "top": 0,    "bottom": 500},   # extended header
    {"name": "mid_300",  "top": 100,  "bottom": 400},   # center (login forms)
]

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
            print(f"[visual_matcher] load error: {e}")
            return False


def is_visual_available() -> bool:
    return EMBEDDINGS_PATH.exists()


def _crop_and_embed(img: Image.Image, top: int, bottom: int) -> Optional[np.ndarray]:
    w, h = img.size
    actual_bottom = min(bottom, h)
    actual_top = min(top, actual_bottom)
    if actual_bottom <= actual_top:
        return None
    crop = img.crop((0, actual_top, w, actual_bottom))
    try:
        tensor = _preprocess(crop).unsqueeze(0)
        with torch.no_grad():
            emb = _model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze().numpy()
    except Exception:
        return None


def _compute_query_embeddings(image_path: str) -> list[np.ndarray]:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[visual_matcher] image load error: {e}")
        return []

    embeddings = []
    for strategy in CROP_STRATEGIES:
        emb = _crop_and_embed(img, strategy["top"], strategy["bottom"])
        if emb is not None:
            embeddings.append(emb)
    return embeddings


def match_brand(screenshot_path: str, threshold: float = 0.80) -> dict:
    no_match = {"matched": False, "brand": None, "display": None,
                "similarity": 0.0, "label": None}

    if not _load():
        return no_match

    query_embeddings = _compute_query_embeddings(screenshot_path)
    if not query_embeddings:
        return no_match

    best_brand = None
    best_display = None
    best_sim = -1.0
    best_label = None

    for brand_name, brand_data in _embeddings.items():
        for ref in brand_data.get("references", []):
            # Support both old format (single embedding) and new format (list of embeddings)
            ref_embeddings = ref.get("embeddings", [])
            if not ref_embeddings:
                single = ref.get("embedding")
                if single is not None:
                    ref_embeddings = [single]

            for ref_emb in ref_embeddings:
                if ref_emb is None:
                    continue
                # Compare each query crop against each reference embedding
                for query_emb in query_embeddings:
                    sim = float(np.dot(query_emb, ref_emb))
                    if sim > best_sim:
                        best_sim = sim
                        best_brand = brand_name
                        best_display = brand_data.get("display", brand_name)
                        best_label = ref.get("label")

    if best_sim >= threshold:
        return {
            "matched": True,
            "brand": best_brand,
            "display": best_display,
            "similarity": round(best_sim, 4),
            "label": best_label,
        }

    return {**no_match, "similarity": round(best_sim, 4)}