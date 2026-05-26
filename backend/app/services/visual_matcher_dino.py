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


def match_brand_dino(screenshot_path: str, threshold: float = 0.80) -> dict:
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