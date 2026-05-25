"""Random Forest inference service for phishing URL detection."""

import os
from threading import Lock
from typing import Optional

import joblib
import numpy as np

from app.services.url_analyzer import extract_features

MODEL_PATH = os.environ.get("RF_MODEL_PATH", "/app/models/rf_model.joblib")

_model = None
_feature_order = None
_lock = Lock()


def _load_model() -> None:
    """Lazy-load the trained model on first request."""
    global _model, _feature_order
    with _lock:
        if _model is not None:
            return
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run `python scripts/train_rf.py` first."
            )
        bundle = joblib.load(MODEL_PATH)
        _model = bundle["model"]
        _feature_order = bundle["feature_order"]


def is_model_available() -> bool:
    return os.path.exists(MODEL_PATH)


def predict_ml_score(url: str) -> dict:
    """Return phishing probability for a single URL.

    Returns:
        dict with keys: score (float in [0,1]), features (dict of 20 features).
    """
    if _model is None:
        _load_model()

    features = extract_features(url)
    vector = np.asarray([[features[k] for k in _feature_order]], dtype=np.float32)
    proba = float(_model.predict_proba(vector)[0, 1])
    return {"score": round(proba, 4), "features": features}