"""
DINO-PX+Filter knowledge-base builder.

Combines the PX crop geometry (top_150, top_300, top_500, mid_300, absolute
pixel coordinates) with the uniform-color filter (grayscale std < 12),
applied at indexing time, using DINOv2 (facebook/dinov2-base).

Reuses BRANDS and capture() from init_brand_db_dino.py — same screenshot set
as DINO-PX, no new Playwright captures needed.

Output: /app/data/brand_embeddings_dino_px_filtered.pkl
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

BRANDS_DIR = Path("/app/screenshots/brands")
EMBEDDINGS_PATH = Path("/app/data/brand_embeddings_dino_px_filtered.pkl")

CROP_STRATEGIES = [
    {"name": "top_150", "top": 0,   "bottom": 150},
    {"name": "top_300", "top": 0,   "bottom": 300},
    {"name": "top_500", "top": 0,   "bottom": 500},
    {"name": "mid_300", "top": 100, "bottom": 400},
]

UNIFORM_STD_THRESHOLD = 12.0


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


def compute_embeddings(model, processor, image_path: str) -> list:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"    [FAIL] image load: {e}")
        return []

    w, h = img.size
    kept, skipped = 0, 0
    embeddings = []

    for strategy in CROP_STRATEGIES:
        top = max(0, min(h, strategy["top"]))
        bottom = max(0, min(h, strategy["bottom"]))
        if bottom <= top:
            continue
        crop = img.crop((0, top, w, bottom))
        if _is_uniform(crop):
            skipped += 1
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
            skipped += 1

    print(f"    [crops] kept={kept}  skipped_uniform={skipped}")
    return embeddings


async def main():
    BRANDS_DIR.mkdir(parents=True, exist_ok=True)
    Path("/app/data").mkdir(parents=True, exist_ok=True)

    embeddings: dict = {}
    if EMBEDDINGS_PATH.exists():
        with open(EMBEDDINGS_PATH, "rb") as f:
            embeddings = pickle.load(f)
        print(f"Resuming: {len(embeddings)} brands already in dino-px-filtered pkl\n")

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
                print(f"  [{label}] skip — already indexed")
                continue

            dest = BRANDS_DIR / f"{name}_{label}.png"
            ok = await capture(ref["url"], dest)
            if not ok:
                await asyncio.sleep(2)
                continue

            embs = compute_embeddings(model, processor, str(dest))
            if not embs:
                print(f"  [{label}] DROPPED — all crops uniform")
                continue

            new_refs.append({
                "label": label,
                "url": ref["url"],
                "screenshot": str(dest),
                "embeddings": embs,
            })
            print(f"  [{label}] OK — {len(embs)} usable crop embeddings")
            changed = True
            await asyncio.sleep(1)

        if changed or name not in embeddings:
            embeddings[name] = {
                "display": brand["display"],
                "category": brand["category"],
                "references": new_refs,
            }
            with open(EMBEDDINGS_PATH, "wb") as f:
                pickle.dump(embeddings, f)

    processed = sum(1 for v in embeddings.values() if v["references"])
    total_refs = sum(len(v["references"]) for v in embeddings.values())
    print(f"\nDone: {processed}/{total} brands, {total_refs} total references")
    print(f"Saved: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())