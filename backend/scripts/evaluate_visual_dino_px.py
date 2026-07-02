import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from app.services.visual_matcher_dino_px_filtered import match_brand_dino_px_filtered, _load

def init_worker():
    _load()

def process_single_brand(brand_dir):
    brand_key = brand_dir.name
    results = {"tp": 0, "fp": 0, "fn": 0, "samples": 0}
    
    sample_dirs = [d for d in brand_dir.iterdir() if d.is_dir()]
    for sample_dir in sample_dirs:
        shot = sample_dir / "shot.png"
        if not shot.exists(): continue
        
        results["samples"] += 1
        try:
            res = match_brand_dino_px_filtered(str(shot))
            if res["matched"]:
                if res["brand"] == brand_key: results["tp"] += 1
                else: results["fp"] += 1
            else: results["fn"] += 1
        except Exception:
            results["fn"] += 1
    return brand_key, results

def evaluate():
    base_path = Path("/app/evaluation/phishpedia_filtered")
    brand_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    with ProcessPoolExecutor(max_workers=2, initializer=init_worker) as executor:
        print(f"Începem evaluarea pentru {len(brand_dirs)} branduri...")
        final_results = dict(executor.map(process_single_brand, brand_dirs))

    output_path = Path("/app/backend/results/visual_evaluation_dino_px_filtered.json")
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Gata! Rezultate salvate în {output_path}")

if __name__ == "__main__":
    evaluate()