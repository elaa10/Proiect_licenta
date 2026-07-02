
import argparse
import os
import re
import sys

import pandas as pd
import numpy as np

VALID_URL_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9.\-]*\.[a-zA-Z]{2,6}(/.*)?$")


def is_clean_url(u) -> bool:
    if not isinstance(u, str):
        return False
    if len(u) < 8 or len(u) > 500:
        return False
    if not all(32 <= ord(c) < 127 for c in u):
        return False
    return bool(VALID_URL_PATTERN.match(u))


def normalize_url(u: str, force_scheme: bool = True) -> str:
    if not force_scheme or u.startswith(("http://", "https://")):
        return u
    use_https = (hash(u) % 10) < 7
    return f"{'https' if use_https else 'http'}://{u}"


def augment_www_variations(df: pd.DataFrame, ratio: float = 0.5, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)

    legit_mask = df["y"] == 0
    legit_idx = df.index[legit_mask].tolist()
    n_swap = int(len(legit_idx) * ratio)
    to_swap = rng.choice(legit_idx, size=n_swap, replace=False)

    swapped = 0
    for idx in to_swap:
        url = df.at[idx, "url"]
        if "://www." in url:
            new_url = url.replace("://www.", "://", 1)
        else:
            for scheme in ("http://", "https://"):
                if url.startswith(scheme):
                    rest = url[len(scheme):]
                    if not rest.startswith("www."):
                        new_url = f"{scheme}www.{rest}"
                        break
            else:
                continue
        df.at[idx, "url"] = new_url
        swapped += 1
    print(f"[augment] Swapped www. for {swapped}/{len(legit_idx)} legitimate URLs ({100*swapped/len(legit_idx):.1f}%)")
    return df


def load_phiusiil(csv_path: str) -> pd.DataFrame:
    print(f"[phiusiil] Loading {csv_path}")
    df = pd.read_csv(csv_path)[["URL", "label"]]
    df.columns = ["url", "label_raw"]
    df["y"] = (df["label_raw"] == 0).astype(int)
    df = df.drop(columns=["label_raw"])
    print(f"[phiusiil]   total={len(df)}  phishing={int(df['y'].sum())}  legitimate={int((df['y']==0).sum())}")
    return df


def load_kaggle(csv_path: str) -> pd.DataFrame:
    print(f"[kaggle] Loading {csv_path}")
    df = pd.read_csv(csv_path)[["URL", "Label"]]
    df.columns = ["url", "label_raw"]
    df["y"] = (df["label_raw"].astype(str).str.lower() == "bad").astype(int)
    df = df.drop(columns=["label_raw"])

    before = len(df)
    df = df[df["url"].apply(is_clean_url)].reset_index(drop=True)
    print(f"[kaggle]   filtered: {len(df)}/{before} ({100*len(df)/before:.1f}% retained)")

    df["url"] = df["url"].apply(lambda u: normalize_url(u, force_scheme=True))
    print(f"[kaggle]   phishing={int(df['y'].sum())}  legitimate={int((df['y']==0).sum())}")
    return df


def build_balanced_sample(
    df_phi: pd.DataFrame,
    df_kag: pd.DataFrame,
    n_per_class: int,
    phi_phish_ratio: float = 0.6,
    phi_legit_ratio: float = 0.5,
    random_state: int = 42,
) -> pd.DataFrame:

    rng = np.random.RandomState(random_state)

    phi_phish = df_phi[df_phi["y"] == 1]
    phi_legit = df_phi[df_phi["y"] == 0]
    kag_phish = df_kag[df_kag["y"] == 1]
    kag_legit = df_kag[df_kag["y"] == 0]

    # Phishing
    n_phi_phish = min(int(n_per_class * phi_phish_ratio), len(phi_phish))
    n_kag_phish = min(n_per_class - n_phi_phish, len(kag_phish))
    # Daca PhiUSIIL nu are destule, completam cu Kaggle
    if n_phi_phish + n_kag_phish < n_per_class:
        n_kag_phish = min(len(kag_phish), n_per_class - n_phi_phish)

    # Legitim
    n_phi_legit = min(int(n_per_class * phi_legit_ratio), len(phi_legit))
    n_kag_legit = min(n_per_class - n_phi_legit, len(kag_legit))
    if n_phi_legit + n_kag_legit < n_per_class:
        n_kag_legit = min(len(kag_legit), n_per_class - n_phi_legit)

    print(f"\n[mix] Phishing:   PhiUSIIL={n_phi_phish}  + Kaggle={n_kag_phish}  = {n_phi_phish + n_kag_phish}")
    print(f"[mix] Legitimate: PhiUSIIL={n_phi_legit}  + Kaggle={n_kag_legit}  = {n_phi_legit + n_kag_legit}")

    parts = [
        phi_phish.sample(n=n_phi_phish, random_state=rng.randint(1e6)),
        kag_phish.sample(n=n_kag_phish, random_state=rng.randint(1e6)),
        phi_legit.sample(n=n_phi_legit, random_state=rng.randint(1e6)),
        kag_legit.sample(n=n_kag_legit, random_state=rng.randint(1e6)),
    ]
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset="url").sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"[mix] Total dupa shuffle si deduplicare: {len(combined)}  (phish={int(combined['y'].sum())}, legit={int((combined['y']==0).sum())})")
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phiusiil", default="/app/data/PhiUSIIL_Phishing_URL_Dataset.csv")
    parser.add_argument("--kaggle", default="/app/data/phishing_site_urls.csv")
    parser.add_argument("--output", default="/app/data/merged_training_dataset.csv")
    parser.add_argument("--n-per-class", type=int, default=75000,
                        help="URL-uri per clasa (phishing si legitim). Total = 2 * n_per_class.")
    args = parser.parse_args()

    if not os.path.exists(args.phiusiil):
        print(f"ERROR: PhiUSIIL CSV not found at {args.phiusiil}")
        sys.exit(1)
    if not os.path.exists(args.kaggle):
        print(f"ERROR: Kaggle CSV not found at {args.kaggle}")
        sys.exit(1)

    df_phi = load_phiusiil(args.phiusiil)
    df_kag = load_kaggle(args.kaggle)

    merged = build_balanced_sample(df_phi, df_kag, n_per_class=args.n_per_class)
    merged = augment_www_variations(merged, ratio=0.5, random_state=42)

    out = merged[["url", "y"]]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\n[save] Merged dataset written to: {args.output}")
    print(f"[save] {len(out)} rows  phishing={int((out['y']==1).sum())}  legitimate={int((out['y']==0).sum())}")

    print(f"\n=== Lungime URL (verificare bias) ===")
    for y_val, name in [(1, "Phishing"), (0, "Legitim")]:
        lens = out[out["y"] == y_val]["url"].str.len()
        print(f"  {name:<10}  mean={lens.mean():.1f}  median={lens.median():.0f}  min={lens.min()}  max={lens.max()}")


if __name__ == "__main__":
    main()