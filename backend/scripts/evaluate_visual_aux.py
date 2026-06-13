"""
AUX CLIP visual module evaluation on Phishpedia benchmark.

Goes in: backend/scripts/evaluate_visual_aux.py

Output: backend/results/visual_evaluation_aux.json
Compare with: backend/results/visual_evaluation_multicrop.json (current model)
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/app")
from app.services.visual_matcher_aux import match_brand_aux, _load

FILTERED_DIR = Path("/app/evaluation/phishpedia_filtered")
RESULTS_DIR  = Path("/app/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIVE_BRANDS = {
    "paypal", "ing", "microsoft", "facebook", "amazon",
    "netflix", "dhl", "apple", "linkedin", "adobe",
    "dropbox", "instagram", "ebay", "google", "steam", "whatsapp",
}


def evaluate():
    print("Loading AUX CLIP matcher (extended crops + uniform filter)...")
    if not _load():
        print("ERROR: brand_embeddings_aux.pkl not found. "
              "Run scripts/init_brand_db_aux.py first.")
        sys.exit(1)
    print("AUX matcher ready.\n")

    per_brand = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "samples": 0})
    all_sims = []

    brand_dirs = sorted(FILTERED_DIR.iterdir())
    total_brands = len(brand_dirs)

    for b_idx, brand_dir in enumerate(brand_dirs):
        if not brand_dir.is_dir():
            continue

        brand_key = brand_dir.name
        sample_dirs = sorted(brand_dir.iterdir())
        n = len(sample_dirs)
        print(f"[{b_idx+1:02d}/{total_brands}] {brand_key:25s} ({n} samples)")

        for sample_dir in sample_dirs:
            shot = sample_dir / "shot.png"
            if not shot.exists():
                continue

            per_brand[brand_key]["samples"] += 1

            try:
                result = match_brand_aux(str(shot))
            except Exception:
                per_brand[brand_key]["fn"] += 1
                continue

            all_sims.append(result["similarity"])

            if result["matched"]:
                if result["brand"] == brand_key:
                    per_brand[brand_key]["tp"] += 1
                else:
                    per_brand[brand_key]["fp"] += 1
            else:
                per_brand[brand_key]["fn"] += 1

        tp = per_brand[brand_key]["tp"]
        fn = per_brand[brand_key]["fn"]
        fp = per_brand[brand_key]["fp"]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"         TP={tp} FN={fn} FP={fp} recall={recall:.2%}")

    total_tp = sum(v["tp"] for v in per_brand.values())
    total_fp = sum(v["fp"] for v in per_brand.values())
    total_fn = sum(v["fn"] for v in per_brand.values())
    total_samples = sum(v["samples"] for v in per_brand.values())

    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

    per_brand_metrics = {}
    for brand, v in sorted(per_brand.items(), key=lambda x: -x[1]["samples"]):
        tp, fp, fn = v["tp"], v["fp"], v["fn"]
        bp = tp / (tp + fp) if (tp + fp) > 0 else 0
        br = tp / (tp + fn) if (tp + fn) > 0 else 0
        bf = 2 * bp * br / (bp + br) if (bp + br) > 0 else 0
        per_brand_metrics[brand] = {
            "samples": v["samples"], "tp": tp, "fp": fp, "fn": fn,
            "precision": round(bp, 4), "recall": round(br, 4), "f1": round(bf, 4),
            "representative": brand in REPRESENTATIVE_BRANDS,
        }

    results = {
        "model": "CLIP ViT-B/32 — AUX (9 crops + uniform-color filter)",
        "strategy": "multi-crop with std-based uniform skip (threshold 12)",
        "dataset": "Phishpedia benchmark (Lin et al., USENIX Security 2021)",
        "total_samples": total_samples,
        "threshold": 0.80,
        "overall": {
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "detection_rate": round(total_tp / total_samples if total_samples else 0, 4),
        },
        "similarity_stats": {
            "mean": round(float(np.mean(all_sims)), 4) if all_sims else 0,
            "std":  round(float(np.std(all_sims)), 4) if all_sims else 0,
        },
        "per_brand": per_brand_metrics,
    }

    out = RESULTS_DIR / "visual_evaluation_aux.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"AUX CLIP RESULTS ({total_samples} samples)")
    print(f"{'='*60}")
    print(f"  Precision : {p:.2%}")
    print(f"  Recall    : {r:.2%}")
    print(f"  F1        : {f1:.2%}")
    print(f"\nResults written to: {out}")
    print(f"Reference: backend/results/visual_evaluation_multicrop.json")


if __name__ == "__main__":
    evaluate()
