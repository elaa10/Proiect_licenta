"""
AUX CLIP knowledge-base builder.

Goes in: backend/scripts/init_brand_db_aux.py

Reuses BRANDS and capture() from the original init_brand_db.py — same brand
list, same Playwright capture logic, same screenshots dir. The only changes:
  - Uses CROP_STRATEGIES and UNIFORM_STD_THRESHOLD from visual_matcher_aux.
  - Skips crops that fail the uniform-color filter at indexing time.
  - Writes to /app/data/brand_embeddings_aux.pkl (separate from main pkl).

Run inside the backend container:
    python scripts/init_brand_db_aux.py
"""
import asyncio
import pickle
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

sys.path.insert(0, "/app")
from scripts.init_brand_db import BRANDS, capture  # noqa: E402
from app.services.visual_matcher_aux import (  # noqa: E402
    CROP_STRATEGIES, UNIFORM_STD_THRESHOLD,
)

BRANDS_DIR = Path("/app/screenshots/brands")
EMBEDDINGS_PATH = Path("/app/data/brand_embeddings_aux.pkl")


def load_clip_model():
    print("Loading CLIP ViT-B/32...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval()
    print("CLIP model ready\n")
    return model, preprocess


def _is_uniform(image: Image.Image) -> bool:
    try:
        arr = np.asarray(image.convert("L"), dtype=np.float32)
        return float(arr.std()) < UNIFORM_STD_THRESHOLD
    except Exception:
        return True


def compute_multi_crop_embeddings(model, preprocess, image_path: str) -> list:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"    [FAIL] image load: {e}")
        return []

    w, h = img.size
    kept, skipped_uniform, skipped_invalid = 0, 0, 0
    embeddings = []

    for strategy in CROP_STRATEGIES:
        left   = max(0, min(w, int(round(strategy["x1"] * w))))
        top    = max(0, min(h, int(round(strategy["y1"] * h))))
        right  = max(0, min(w, int(round(strategy["x2"] * w))))
        bottom = max(0, min(h, int(round(strategy["y2"] * h))))
        if right <= left or bottom <= top:
            skipped_invalid += 1
            continue

        crop = img.crop((left, top, right, bottom))
        if _is_uniform(crop):
            skipped_uniform += 1
            continue

        try:
            tensor = preprocess(crop).unsqueeze(0)
            with torch.no_grad():
                emb = model.encode_image(tensor)
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
        print(f"Resuming: {len(embeddings)} brands already in aux pkl\n")

    model, preprocess = load_clip_model()
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
                print(f"  [{label}] skip — already in aux pkl")
                continue

            # Reuse the screenshot saved by the main init_brand_db.py if present.
            dest = BRANDS_DIR / f"{name}_{label}.png"
            ok = await capture(ref["url"], dest)
            if not ok:
                await asyncio.sleep(2)
                continue

            embs = compute_multi_crop_embeddings(model, preprocess, str(dest))
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
                print(f"  [{label}] DROPPED — all crops were uniform "
                      f"(reference is unusable, add manual screenshot)")

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
