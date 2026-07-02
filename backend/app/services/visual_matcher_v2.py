
import pickle
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import open_clip
import torch
from PIL import Image, ImageChops

EMBEDDINGS_PATH_V2 = Path("/app/data/brand_embeddings_v2.pkl")

# Strategie Hibridă
CROP_STRATEGIES = [
    {"name": "logo_left_tight",   "type": "pixel",        "x": 0, "y": 0, "w": 350, "h": 200},
    {"name": "logo_center_tight", "type": "proportional", "x1": 0.35, "y1": 0.00, "x2": 0.65, "y2": 0.25},
    {"name": "center_login_box",  "type": "proportional", "x1": 0.30, "y1": 0.15, "x2": 0.70, "y2": 0.65},
    {"name": "upper_third_wide",  "type": "proportional", "x1": 0.00, "y1": 0.00, "x2": 1.00, "y2": 0.35},
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
    if not EMBEDDINGS_PATH_V2.exists():
        return False
    with _lock:
        if _model is not None:
            return True
        try:
            _model, _, _preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            _model.eval()
            with open(EMBEDDINGS_PATH_V2, "rb") as f:
                _embeddings = pickle.load(f)
            return True
        except Exception as e:
            print(f"[visual_matcher_v2] load error: {e}")
            return False

def is_visual_available() -> bool:
    return EMBEDDINGS_PATH_V2.exists()

def trim_white_borders(im: Image.Image) -> Optional[Image.Image]:
    bg = Image.new(im.mode, im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    
    if bbox:
        return im.crop(bbox)
    
    return None # Imaginea a fost 100% albă

def _apply_strategy(img: Image.Image, strategy: dict) -> Optional[Image.Image]:
    w, h = img.size
    
    if strategy.get("type") == "pixel":
        left = strategy.get("x", 0)
        top = strategy.get("y", 0)
        right = min(w, left + strategy.get("w", w))
        bottom = min(h, top + strategy.get("h", h))
    else:
        left = int(round(strategy["x1"] * w))
        top = int(round(strategy["y1"] * h))
        right = int(round(strategy["x2"] * w))
        bottom = int(round(strategy["y2"] * h))
        
    left, top = max(0, left), max(0, top)
    
    if right <= left or bottom <= top:
        return None
        
    crop = img.crop((left, top, right, bottom))
    
    return trim_white_borders(crop)

def _embed_crop(crop: Image.Image) -> Optional[np.ndarray]:
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
        print(f"[visual_matcher_v2] image load error: {e}")
        return []

    embeddings = []
    for strategy in CROP_STRATEGIES:
        crop = _apply_strategy(img, strategy)
        if crop is None:
            continue
            
        emb = _embed_crop(crop)
        if emb is not None:
            embeddings.append(emb)
    return embeddings

def match_brand(screenshot_path: str, threshold: float = 0.85) -> dict:
    print(f" ---> ANALIZEZ FOLOSIND V2 BAZA DE DATE: {EMBEDDINGS_PATH_V2}")
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
            "display": top_info['display'],
            "similarity": round(top_info["similarity"], 4),
            "label": top_info["label"],
        }

    return {**no_match, "similarity": round(top_info["similarity"], 4)}