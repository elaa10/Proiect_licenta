
import json
import sys
import concurrent.futures
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/app")

from app.services.visual_matcher_v2 import match_brand, _load

FILTERED_DIR = Path("/app/evaluation/phishpedia_filtered")
RESULTS_DIR  = Path("/app/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIVE_BRANDS = {
    "paypal", "ing", "microsoft", "facebook", "amazon",
    "netflix", "dhl", "apple", "linkedin", "adobe",
    "dropbox", "instagram", "ebay", "google", "steam", "whatsapp",
}

MAX_WORKERS = 8

def _process_single_sample(brand_key, sample_dir):
    shot = sample_dir / "shot.png"
    if not shot.exists():
        return brand_key, sample_dir.name, None, "File not found"
    
    try:
        result = match_brand(str(shot))
        return brand_key, sample_dir.name, result, None
    except Exception as e:
        return brand_key, sample_dir.name, None, str(e)


def evaluate():
    print("Loading CLIP model and V2 embeddings...")
    if not _load():
        print("ERROR: Could not load visual matcher V2. Run init_brand_db_v2.py first.")
        sys.exit(1)
    print("Model ready. Starting parallel evaluation...\n")

    per_brand = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "samples": 0, "correct_brand": 0})
    all_similarities = []
    errors = []

    brand_dirs = sorted(FILTERED_DIR.iterdir())
    total_brands = len(brand_dirs)

    for b_idx, brand_dir in enumerate(brand_dirs):
        if not brand_dir.is_dir():
            continue

        brand_key = brand_dir.name
        sample_dirs = sorted(brand_dir.iterdir())
        n = len(sample_dirs)
        print(f"[{b_idx+1:02d}/{total_brands}] {brand_key:25s} ({n} samples) - Processing...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(_process_single_sample, brand_key, s_dir) for s_dir in sample_dirs]
            
            for future in concurrent.futures.as_completed(futures):
                b_key, s_name, result, err = future.result()
                
                if err == "File not found":
                    continue
                    
                per_brand[brand_key]["samples"] += 1

                if err:
                    errors.append({"brand": brand_key, "sample": s_name, "error": err})
                    per_brand[brand_key]["fn"] += 1
                    continue

                sim = result["similarity"]
                all_similarities.append(sim)

                if result["matched"]:
                    if result["brand"] == brand_key:
                        per_brand[brand_key]["tp"] += 1
                        per_brand[brand_key]["correct_brand"] += 1
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

    rep_tp = sum(v["tp"] for k, v in per_brand.items() if k in REPRESENTATIVE_BRANDS)
    rep_fp = sum(v["fp"] for k, v in per_brand.items() if k in REPRESENTATIVE_BRANDS)
    rep_fn = sum(v["fn"] for k, v in per_brand.items() if k in REPRESENTATIVE_BRANDS)
    rep_samples = sum(v["samples"] for k, v in per_brand.items() if k in REPRESENTATIVE_BRANDS)
    rep_precision = rep_tp / (rep_tp + rep_fp) if (rep_tp + rep_fp) > 0 else 0
    rep_recall    = rep_tp / (rep_tp + rep_fn) if (rep_tp + rep_fn) > 0 else 0
    rep_f1        = 2 * rep_precision * rep_recall / (rep_precision + rep_recall) if (rep_precision + rep_recall) > 0 else 0

    per_brand_metrics = {}
    for brand, v in sorted(per_brand.items(), key=lambda x: -x[1]["samples"]):
        tp, fp, fn = v["tp"], v["fp"], v["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        id_rate = v["correct_brand"] / v["samples"] if v["samples"] > 0 else 0
        per_brand_metrics[brand] = {
            "samples": v["samples"],
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "identification_rate": round(id_rate, 4),
            "representative": brand in REPRESENTATIVE_BRANDS,
        }

    results = {
        "dataset": "Phishpedia benchmark (Lin et al., USENIX Security 2021) - V2 Evaluation",
        "total_samples": total_samples,
        "total_brands": total_brands,
        "threshold": 0.85, 
        "strategy": "Hybrid (Pixel + Proportional) with White Border Trimming",
        "overall": {
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "detection_rate": round(detection_rate, 4),
        },
        "representative_brands_only": {
            "brands": sorted(REPRESENTATIVE_BRANDS),
            "samples": rep_samples,
            "precision": round(rep_precision, 4),
            "recall": round(rep_recall, 4),
            "f1": round(rep_f1, 4),
        },
        "similarity_stats": {
            "mean": round(float(np.mean(all_similarities)), 4) if all_similarities else 0.0,
            "std":  round(float(np.std(all_similarities)), 4) if all_similarities else 0.0,
            "min":  round(float(np.min(all_similarities)), 4) if all_similarities else 0.0,
            "max":  round(float(np.max(all_similarities)), 4) if all_similarities else 0.0,
        },
        "per_brand": per_brand_metrics,
        "errors": errors[:20],
    }

    out_path = RESULTS_DIR / "visual_evaluation_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print(f"V2 OVERALL RESULTS ({total_samples} samples, {total_brands} brands)")
    print(f"{'='*55}")
    print(f"  Precision      : {precision:.2%}")
    print(f"  Recall         : {recall:.2%}")
    print(f"  F1 Score       : {f1:.2%}")
    print(f"  Detection Rate : {detection_rate:.2%}")
    print(f"\nREPRESENTATIVE BRANDS ONLY (≥100 samples)")
    print(f"  Precision      : {rep_precision:.2%}")
    print(f"  Recall         : {rep_recall:.2%}")
    print(f"  F1 Score       : {rep_f1:.2%}")
    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    evaluate()