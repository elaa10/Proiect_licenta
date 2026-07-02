
import pickle
import numpy as np

with open("/app/data/brand_embeddings_px_filtered.pkl", "rb") as f:
    emb = pickle.load(f)

BRANDS_TO_CHECK = ["google", "youtube", "paypal", "anaf", "roeid"]

vecs = {}
for name in BRANDS_TO_CHECK:
    if name not in emb:
        print(f"{name}: NOT FOUND in pkl")
        continue
    for r in emb[name]["references"]:
        if r["label"] == "home":
            vecs[name] = r["embeddings"]
            print(f"{name}: {len(r['embeddings'])} crop embeddings, "
                  f"screenshot={r.get('screenshot')}")

print()
keys = list(vecs.keys())

for crop_idx in range(4):
    print(f"--- crop {crop_idx} ---")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            brand_a = keys[i]
            brand_b = keys[j]
            
            if crop_idx < len(vecs[brand_a]) and crop_idx < len(vecs[brand_b]):
                a = vecs[brand_a][crop_idx]
                b = vecs[brand_b][crop_idx]
                
                if a is not None and b is not None:
                    sim = float(np.dot(a, b))
                    print(f"  {brand_a:<12} vs {brand_b:<12}: {sim:.6f}")
                else:
                    print(f"  {brand_a:<12} vs {brand_b:<12}: SKIP (filtru activat)")
            else:
                print(f"  {brand_a:<12} vs {brand_b:<12}: SKIP (lipsa crop, filtru activat)")