"""
AUX DINOv2 knowledge-base builder.

Goes in: backend/scripts/init_brand_db_dino_aux.py

Same as init_brand_db_aux.py but uses DINOv2 instead of CLIP.
Writes to /app/data/brand_embeddings_dino_aux.pkl.
"""
import asyncio
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/app")
from scripts.init_brand_db_dino import BRANDS, capture  # noqa: E402
from app.services.visual_matcher_dino_aux import (  # noqa: E402
    CROP_STRATEGIES, UNIFORM_STD_THRESHOLD,
)

BRANDS_DIR = Path("/app/screenshots/brands")
EMBEDDINGS_PATH = Path("/app/data/brand_embeddings_dino_aux.pkl")


def load_dino_model():
    print("Loading DINOv2 (facebook/dinov2-base)...")
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model.eval()
    print("DINOv2 model ready\n")
    return model, processor


def _is_uniform(image: Image.Image) -> bool:
    try:
        arr = np.asarray(image.convert("L"), dtype=np.float32)
        return float(arr.std()) < UNIFORM_STD_THRESHOLD
    except Exception:
        return True


def compute_multi_crop_embeddings(model, processor, image_path: str) -> list:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"    [FAIL] image load: {e}")
        return []

    w, h = img.size
    kept, skipped_uniform = 0, 0
    embeddings = []

    for strategy in CROP_STRATEGIES:
        left   = max(0, min(w, int(round(strategy["x1"] * w))))
        top    = max(0, min(h, int(round(strategy["y1"] * h))))
        right  = max(0, min(w, int(round(strategy["x2"] * w))))
        bottom = max(0, min(h, int(round(strategy["y2"] * h))))
        if right <= left or bottom <= top:
            continue

        crop = img.crop((left, top, right, bottom))
        if _is_uniform(crop):
            skipped_uniform += 1
            continue

        try:
            inputs = processor(images=crop, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :]
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.squeeze().numpy())
            kept += 1
        except Exception:
            continue

    print(f"    [crops] kept={kept}  skipped_uniform={skipped_uniform}")
    return embeddings


async def main():
    BRANDS_DIR.mkdir(parents=True, exist_ok=True)
    Path("/app/data").mkdir(parents=True, exist_ok=True)

    embeddings: dict = {}
    if EMBEDDINGS_PATH.exists():
        with open(EMBEDDINGS_PATH, "rb") as f:
            embeddings = pickle.load(f)
        print(f"Resuming: {len(embeddings)} brands already in dino aux pkl\n")

    model, processor = load_dino_model()
    total = len(BRANDS)

    for i, brand in enumerate(BRANDS):
        name = brand["name"]
        refs = brand["references"]
        print(f"[{i+1:02d}/{total}] {brand['display']} ({len(refs)} reference(s))")

        existing_refs = embeddings.get(name, {}).get("references", [])
        existing_labels = {r["label"] for r in existing_refs}
        new_refs = list(existing_refs)
        changed = False

        for ref in refs:
            label = ref["label"]
            if label in existing_labels:
                print(f"  [{label}] skip")
                continue

            dest = BRANDS_DIR / f"{name}_{label}.png"
            ok = await capture(ref["url"], dest)
            if not ok:
                await asyncio.sleep(2)
                continue

            embs = compute_multi_crop_embeddings(model, processor, str(dest))
            if embs:
                new_refs.append({
                    "label": label,
                    "url": ref["url"],
                    "screenshot": str(dest),
                    "embeddings": embs,
                })
                print(f"  [{label}] OK — {len(embs)} usable crop embeddings")
                changed = True
            else:
                print(f"  [{label}] DROPPED — all crops were uniform")

            await asyncio.sleep(1)

        if changed or name not in embeddings:
            embeddings[name] = {
                "display": brand["display"],
                "category": brand["category"],
                "references": new_refs,
            }
            with open(EMBEDDINGS_PATH, "wb") as f:
                pickle.dump(embeddings, f)

    total_refs = sum(len(v["references"]) for v in embeddings.values())
    total_embs = sum(
        sum(len(r.get("embeddings", [])) for r in v["references"])
        for v in embeddings.values()
    )
    print(f"\nDone: {len(embeddings)} brands, {total_refs} references, "
          f"{total_embs} usable embeddings")
    print(f"Saved to: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
