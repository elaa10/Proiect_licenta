import pickle
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import open_clip
import torch
from PIL import Image

EMBEDDINGS_PATH = Path("/app/data/brand_embeddings_aux.pkl")

CROP_STRATEGIES = [
    # ── Originals ──────────────────────────────────────────────────────────
    {"name": "logo_left",          "x1": 0.00, "y1": 0.00, "x2": 0.35, "y2": 0.30},
    {"name": "logo_center",        "x1": 0.25, "y1": 0.00, "x2": 0.75, "y2": 0.35},
    {"name": "upper_third",        "x1": 0.00, "y1": 0.00, "x2": 1.00, "y2": 0.35},
    {"name": "center_band",        "x1": 0.00, "y1": 0.20, "x2": 1.00, "y2": 0.65},
    # ── New, narrower, logo-focused crops ──────────────────────────────────
    {"name": "logo_corner_tl",     "x1": 0.00, "y1": 0.00, "x2": 0.20, "y2": 0.15},
    {"name": "logo_corner_tl_wide","x1": 0.00, "y1": 0.00, "x2": 0.40, "y2": 0.20},
    {"name": "logo_top_strip",     "x1": 0.00, "y1": 0.00, "x2": 1.00, "y2": 0.15},
    {"name": "logo_top_center",    "x1": 0.30, "y1": 0.00, "x2": 0.70, "y2": 0.20},
    {"name": "logo_mid_center",    "x1": 0.25, "y1": 0.15, "x2": 0.75, "y2": 0.45},
]

UNIFORM_STD_THRESHOLD = 12.0

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
            print(f"[visual_matcher_aux] load error: {e}")
            return False


def is_aux_available() -> bool:
    return EMBEDDINGS_PATH.exists()


def _is_uniform(image: Image.Image, threshold: float = UNIFORM_STD_THRESHOLD) -> bool:
    try:
        arr = np.asarray(image.convert("L"), dtype=np.float32)
        return float(arr.std()) < threshold
    except Exception:
        return True


def _crop_proportional(
    img: Image.Image,
    x1: float, y1: float, x2: float, y2: float,
) -> Optional[Image.Image]:
    w, h = img.size
    left   = max(0, min(w, int(round(x1 * w))))
    top    = max(0, min(h, int(round(y1 * h))))
    right  = max(0, min(w, int(round(x2 * w))))
    bottom = max(0, min(h, int(round(y2 * h))))
    if right <= left or bottom <= top:
        return None
    return img.crop((left, top, right, bottom))


def _embed_crop(crop: Image.Image) -> Optional[np.ndarray]:
    if _is_uniform(crop):
        return None
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
        print(f"[visual_matcher_aux] image load error: {e}")
        return []

    embeddings = []
    for strategy in CROP_STRATEGIES:
        crop = _crop_proportional(
            img,
            x1=strategy["x1"], y1=strategy["y1"],
            x2=strategy["x2"], y2=strategy["y2"],
        )
        if crop is None:
            continue
        emb = _embed_crop(crop)
        if emb is not None:
            embeddings.append(emb)
    return embeddings


def match_brand_aux(screenshot_path: str, threshold: float = 0.85) -> dict:
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
