"""
Diagnostic script — DINO-PX+Filter, live captures.
Checks whether the crop_0 (top_150) embedding-collapse phenomenon observed
with CLIP (Section 4.4) also occurs with DINOv2.

Usage: python scripts/diagnose_visual_dino_px_filtered.py <URL> [<URL2> ...]
"""
import asyncio
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/app")

from app.services import visual_matcher_dino_px_filtered as matcher
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

    if not matcher._load():
        print(f"    LOAD FAILED -- {matcher.EMBEDDINGS_PATH} not available")
        return

    embeddings_db = matcher._embeddings
    query_embeds = matcher._compute_query_embeddings(str(path))
    print(f"    Generated {len(query_embeds)} usable crop embeddings "
          f"(out of {len(matcher.CROP_STRATEGIES)})")

    if not query_embeds:
        print("    All crops uniform -- no match possible.")
        return

    scores: dict[str, dict] = {}
    for brand, data in embeddings_db.items():
        max_sim = 0.0
        best_ref_label = None
        for ref in data.get("references", []):
            ref_embeddings = ref.get("embeddings", [])
            for ref_emb in ref_embeddings:
                if ref_emb is None:
                    continue
                for q_emb in query_embeds:
                    sim = float(np.dot(q_emb, ref_emb))
                    if sim > max_sim:
                        max_sim = sim
                        best_ref_label = ref.get("label")
        scores[brand] = {"similarity": max_sim, "reference": best_ref_label}

    top10 = sorted(scores.items(), key=lambda x: -x[1]["similarity"])[:10]
    threshold = 0.85

    print(f"\n{'Rank':<5} {'Brand':<20} {'Similarity':<12} {'Best reference':<15}")
    print("-" * 60)
    for rank, (brand, info) in enumerate(top10, start=1):
        marker = "  <-- MATCH" if rank == 1 and info["similarity"] >= threshold else ""
        print(f"{rank:<5} {brand:<20} {info['similarity']:<12.4f} {info['reference'] or '-':<15}{marker}")

    top1_brand, top1_info = top10[0]
    top2_brand, top2_info = top10[1]
    margin = top1_info["similarity"] - top2_info["similarity"]
    matched = top1_info["similarity"] >= threshold and margin >= matcher.MIN_CONFIDENCE_MARGIN

    print(f"\nTop 1: {top1_brand} ({top1_info['similarity']:.4f})")
    print(f"Top 2: {top2_brand} ({top2_info['similarity']:.4f})")
    print(f"Margin: {margin:.4f}  Matched: {'YES' if matched else 'NO'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_visual_dino_px_filtered.py <URL> [<URL2> ...]")
        sys.exit(1)
    for url in sys.argv[1:]:
        asyncio.run(diagnose(url))