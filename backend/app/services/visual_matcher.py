import pickle
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import open_clip
import torch
from PIL import Image

EMBEDDINGS_PATH = Path("/app/data/brand_embeddings.pkl")
CROP_HEIGHT = 300

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


def _compute_embedding(image_path: str) -> Optional[np.ndarray]:
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        img = img.crop((0, 0, w, min(CROP_HEIGHT, h)))
        tensor = _preprocess(img).unsqueeze(0)
        with torch.no_grad():
            emb = _model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze().numpy()
    except Exception as e:
        print(f"[visual_matcher] embedding error: {e}")
        return None


def match_brand(screenshot_path: str, threshold: float = 0.80) -> dict:
    """Match a screenshot against the brand knowledge base.

    Returns a dict with keys:
        matched (bool), brand (str|None), display (str|None),
        similarity (float), label (str|None)
    """
    no_match = {"matched": False, "brand": None, "display": None,
                "similarity": 0.0, "label": None}

    if not _load():
        return no_match

    query_emb = _compute_embedding(screenshot_path)
    if query_emb is None:
        return no_match

    best_brand = None
    best_display = None
    best_sim = -1.0
    best_label = None

    for brand_name, brand_data in _embeddings.items():
        for ref in brand_data.get("references", []):
            ref_emb = ref.get("embedding")
            if ref_emb is None:
                continue
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