"""
Diagnostic script v2 — shows per-crop, per-strategy similarity breakdown.
Helps identify which specific crop is causing saturation across brands.

Usage: python scripts/diagnose_visual_v2.py <URL>
"""
import asyncio
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/app")

from app.services import visual_matcher
from app.services.browser_capture import capture_screenshot


async def diagnose(url: str) -> None:
    print(f"\n{'='*70}")
    print(f"URL: {url}")
    print(f"{'='*70}")

    filename = await capture_screenshot(url)
    if not filename:
        print("    CAPTURE FAILED")
        return

    path = Path("/app/screenshots") / filename
    print(f"    Screenshot: {path}")

    if not visual_matcher._load():
        print("    LOAD FAILED")
        return

    embeddings_db = visual_matcher._embeddings
    query_embeds = visual_matcher._compute_query_embeddings(str(path))
    strategy_names = [s["name"] for s in visual_matcher.CROP_STRATEGIES]

    print(f"    Crops: {len(query_embeds)} -- {strategy_names}")

    # For each query crop, find top brand
    print(f"\n{'='*70}")
    print(f"PER-CROP SIMILARITY (which crop matches which brand?)")
    print(f"{'='*70}")

    for crop_idx, q_emb in enumerate(query_embeds):
        crop_name = strategy_names[crop_idx]
        # Compute similarity to all brand references for THIS crop only
        scores = {}
        for brand, data in embeddings_db.items():
            best = -1.0
            for ref in data.get("references", []):
                ref_embs = ref.get("embeddings", [])
                if not ref_embs and "embedding" in ref:
                    ref_embs = [ref["embedding"]]
                for r_emb in ref_embs:
                    if r_emb is None:
                        continue
                    s = float(np.dot(q_emb, r_emb))
                    if s > best:
                        best = s
            scores[brand] = best

        top5 = sorted(scores.items(), key=lambda x: -x[1])[:5]
        print(f"\n  Query crop '{crop_name}':")
        for brand, sim in top5:
            print(f"    {brand:<20} {sim:.4f}")

    # Aggregate (current behavior)
    print(f"\n{'='*70}")
    print(f"AGGREGATE (max across all crops, current matcher behavior)")
    print(f"{'='*70}")
    per_brand = {}
    for brand, data in embeddings_db.items():
        best = -1.0
        for ref in data.get("references", []):
            ref_embs = ref.get("embeddings", [])
            if not ref_embs and "embedding" in ref:
                ref_embs = [ref["embedding"]]
            for r_emb in ref_embs:
                if r_emb is None:
                    continue
                for q_emb in query_embeds:
                    s = float(np.dot(q_emb, r_emb))
                    if s > best:
                        best = s
        per_brand[brand] = best

    top10 = sorted(per_brand.items(), key=lambda x: -x[1])[:10]
    for i, (brand, sim) in enumerate(top10, 1):
        print(f"    {i:>2}  {brand:<20} {sim:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_visual_v2.py <URL>")
        sys.exit(1)
    for url in sys.argv[1:]:
        asyncio.run(diagnose(url))