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

    print("\n[1/4] Capturing screenshot via Playwright...")
    filename = await capture_screenshot(url)
    if not filename:
        print("    CAPTURE FAILED -- Playwright returned None")
        return

    path = Path("/app/screenshots") / filename
    print(f"    Screenshot: {path}")
    print(f"    File size:  {path.stat().st_size:,} bytes")

    img = Image.open(path)
    print(f"    Dimensions: {img.size[0]} x {img.size[1]} px")

    print("\n[2/4] Loading embeddings knowledge base...")
    if not visual_matcher._load():
        print("    LOAD FAILED -- brand_embeddings.pkl not available")
        return
    embeddings_db = visual_matcher._embeddings
    print(f"    Loaded {len(embeddings_db)} brands")

    print("\n[3/4] Computing query embeddings (multi-crop)...")
    query_embeds = visual_matcher._compute_query_embeddings(str(path))
    print(f"    Generated {len(query_embeds)} crop embeddings")

    print("\n[4/4] Computing similarity against all brands...")
    scores: dict[str, dict] = {}
    for brand, data in embeddings_db.items():
        max_sim = 0.0
        best_ref_label = None
        for ref in data.get("references", []):
            ref_embeddings = ref.get("embeddings", [])
            if not ref_embeddings:
                single = ref.get("embedding")
                if single is not None:
                    ref_embeddings = [single]
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

    print(f"\n{'='*70}")
    print(f"TOP 10 CANDIDATES")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Brand':<20} {'Similarity':<12} {'Best reference':<15}")
    print("-" * 70)
    for rank, (brand, info) in enumerate(top10, start=1):
        marker = "  <-- MATCH" if rank == 1 and info["similarity"] >= 0.80 else ""
        print(f"{rank:<5} {brand:<20} {info['similarity']:<12.4f} {info['reference'] or '-':<15}{marker}")

    top1_brand, top1_info = top10[0]
    top2_brand, top2_info = top10[1]
    margin = top1_info["similarity"] - top2_info["similarity"]

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")
    print(f"Top 1:        {top1_brand} ({top1_info['similarity']:.4f})")
    print(f"Top 2:        {top2_brand} ({top2_info['similarity']:.4f})")
    print(f"Margin:       {margin:.4f}")
    print(f"Threshold:    0.8000")
    print(f"Matched:      {'YES' if top1_info['similarity'] >= 0.80 else 'NO'}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_visual.py <URL> [<URL2> ...]")
        print("Example: python scripts/diagnose_visual.py https://www.instagram.com/")
        sys.exit(1)

    for url in sys.argv[1:]:
        asyncio.run(diagnose(url))