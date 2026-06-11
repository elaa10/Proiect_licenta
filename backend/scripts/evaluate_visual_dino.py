"""
DINOv2 visual module evaluation on Phishpedia benchmark.
Output: backend/results/visual_evaluation_dino.json
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/app")
from app.services.visual_matcher_dino import match_brand_dino, _load

FILTERED_DIR = Path("/app/evaluation/phishpedia_filtered")
RESULTS_DIR  = Path("/app/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIVE_BRANDS = {
    "paypal", "ing", "microsoft", "facebook", "amazon",
    "netflix", "dhl", "apple", "linkedin", "adobe",
    "dropbox","instagram", "ebay", "google", "steam", "whatsapp",
}

def evaluate():
    print("Loading DINOv2 model and embeddings...")
    if not _load():
        print("ERROR: Run scripts/init_brand_db_dino.py first.")
        sys.exit(1)
    print("Model ready.\n")

    per_brand = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "samples": 0})
    all_similarities = []

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
                result = match_brand_dino(str(shot))
            except Exception:
                per_brand[brand_key]["fn"] += 1
                continue

            all_similarities.append(result["similarity"])
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

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    detection_rate = total_tp / total_samples if total_samples > 0 else 0

    per_brand_metrics = {}
    for brand, v in sorted(per_brand.items(), key=lambda x: -x[1]["samples"]):
        tp, fp, fn = v["tp"], v["fp"], v["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_brand_metrics[brand] = {
            "samples": v["samples"], "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "representative": brand in REPRESENTATIVE_BRANDS,
        }

    results = {
        "model": "DINOv2 (facebook/dinov2-base)",
        "strategy": "multi-crop (top_150, top_300, top_500, mid_300)",
        "dataset": "Phishpedia benchmark (Lin et al., USENIX Security 2021)",
        "total_samples": total_samples,
        "threshold": 0.80,
        "overall": {
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "detection_rate": round(detection_rate, 4),
        },
        "similarity_stats": {
            "mean": round(float(np.mean(all_similarities)), 4),
            "std":  round(float(np.std(all_similarities)), 4),
        },
        "per_brand": per_brand_metrics,
    }

    out_path = RESULTS_DIR / "visual_evaluation_dino.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print(f"DINOV2 RESULTS ({total_samples} samples)")
    print(f"{'='*55}")
    print(f"  Precision      : {precision:.2%}")
    print(f"  Recall         : {recall:.2%}")
    print(f"  F1 Score       : {f1:.2%}")
    print(f"  Detection Rate : {detection_rate:.2%}")
    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    evaluate()