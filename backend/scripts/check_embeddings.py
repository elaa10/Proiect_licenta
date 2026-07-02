
import pickle
import numpy as np

with open("/app/data/brand_embeddings_px.pkl", "rb") as f:
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
            a, b = vecs[keys[i]][crop_idx], vecs[keys[j]][crop_idx]
            sim = float(np.dot(a, b))
            print(f"  {keys[i]:<12} vs {keys[j]:<12}: {sim:.6f}")