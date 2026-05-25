"""
Pregatire dataset combinat pentru antrenarea Random Forest.

Combina doua surse complementare:
  - PhiUSIIL (UCI, 2024): URL-uri recente cu scheme http(s)://, dar URL-uri
    legitime sunt aproape exclusiv homepage-uri scurte (medie 27 caractere).
  - Kaggle Phishing Site URLs (2018): URL-uri legitime cu path-uri reale si
    lungimi variate (medie 45 caractere), dar fara scheme.

Combinarea rezolva bias-ul de lungime care apare daca antrenam doar pe
PhiUSIIL ("URL scurt cu www. = legit, URL cu path = phishing"). Modelul
trebuie sa vada URL-uri legitime de toate lungimile pentru a generaliza.

Output: /app/data/merged_training_dataset.csv cu coloanele (url, y),
unde y=1 pentru phishing, y=0 pentru legitim.
"""

import argparse
import os
import re
import sys

import pandas as pd
import numpy as np

# Filtru de validare URL: trebuie sa arate ca un URL real, nu gunoi binar
VALID_URL_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9.\-]*\.[a-zA-Z]{2,6}(/.*)?$")


def is_clean_url(u) -> bool:
    """Filtreaza URL-urile gibberish/binare din Kaggle (cca 0.2% din date)."""
    if not isinstance(u, str):
        return False
    if len(u) < 8 or len(u) > 500:
        return False
    # Toate caracterele trebuie sa fie ASCII printabile
    if not all(32 <= ord(c) < 127 for c in u):
        return False
    return bool(VALID_URL_PATTERN.match(u))


def normalize_url(u: str, force_scheme: bool = True) -> str:
    """
    Adauga scheme http(s):// daca lipseste, alegand aleator intre http si https.

    Distributia 70/30 (https/http) reflecta tendinta reala observata in trafic:
    ~70% din URL-urile legitime moderne folosesc HTTPS, fara sa ne legam
    100% de un singur protocol (ceea ce ar crea bias spre is_https=1).
    """
    if not force_scheme or u.startswith(("http://", "https://")):
        return u
    # Folosim hash determinist pentru reproductibilitate
    use_https = (hash(u) % 10) < 7
    return f"{'https' if use_https else 'http'}://{u}"


def augment_www_variations(df: pd.DataFrame, ratio: float = 0.5, random_state: int = 42) -> pd.DataFrame:
    """
    Diversifica distributia 'www.' in setul de URL-uri legitime.

    PhiUSIIL legitime sunt 100% cu 'www.', Kaggle legitime sunt 0% cu 'www.'.
    Daca modelul vede 'www.' doar in URL-uri de un anumit tip, va invata fals
    ca 'www.' este semnal de phishing/legit. Solutia: pentru o fractie din
    URL-urile legitime, schimbam 'www.X' cu 'X' si invers.

    Nu aplicam pe phishing pentru a nu schimba semnalele reale.
    """
    rng = np.random.RandomState(random_state)

    legit_mask = df["y"] == 0
    legit_idx = df.index[legit_mask].tolist()
    # Alegem aleator o fractie din URL-urile legitime pentru augmentare
    n_swap = int(len(legit_idx) * ratio)
    to_swap = rng.choice(legit_idx, size=n_swap, replace=False)

    swapped = 0
    for idx in to_swap:
        url = df.at[idx, "url"]
        # Toggle www.: daca exista, scoatem; daca nu exista, adaugam
        if "://www." in url:
            new_url = url.replace("://www.", "://", 1)
        else:
            # Adaugam www. doar daca scheme-ul este detectat
            for scheme in ("http://", "https://"):
                if url.startswith(scheme):
                    rest = url[len(scheme):]
                    # Adaugam www. doar daca rest-ul nu are subdomeniu deja
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
    # PhiUSIIL: label=1 = legitimate, label=0 = phishing
    df["y"] = (df["label_raw"] == 0).astype(int)
    df = df.drop(columns=["label_raw"])
    print(f"[phiusiil]   total={len(df)}  phishing={int(df['y'].sum())}  legitimate={int((df['y']==0).sum())}")
    return df


def load_kaggle(csv_path: str) -> pd.DataFrame:
    print(f"[kaggle] Loading {csv_path}")
    df = pd.read_csv(csv_path)[["URL", "Label"]]
    df.columns = ["url", "label_raw"]
    # Kaggle: 'bad' = phishing, 'good' = legitimate
    df["y"] = (df["label_raw"].astype(str).str.lower() == "bad").astype(int)
    df = df.drop(columns=["label_raw"])

    # Filtram URL-urile gibberish
    before = len(df)
    df = df[df["url"].apply(is_clean_url)].reset_index(drop=True)
    print(f"[kaggle]   filtered: {len(df)}/{before} ({100*len(df)/before:.1f}% retained)")

    # Adaugam scheme la URL-urile Kaggle (toate sunt fara scheme)
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
    """
    Construieste un dataset echilibrat din ambele surse.

    Pentru fiecare clasa (phishing si legitim), luam mix-ul:
      - phi_phish_ratio din PhiUSIIL phishing + restul din Kaggle phishing
      - phi_legit_ratio din PhiUSIIL legitim + restul din Kaggle legitim

    Mix-ul 60% phishing PhiUSIIL este motivat de calitatea mai buna a
    URL-urilor PhiUSIIL (recente, 2024). Mix-ul 50-50 pentru legitim
    rezolva bias-ul de lungime al PhiUSIIL prin URL-uri cu path-uri din Kaggle.
    """
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
    # Shuffle final si deduplicare pe URL
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

    # Salvam cu coloana 'y' (NU 'label'!) ca sa nu se confunde cu schema PhiUSIIL
    # in train_rf.py, unde label=1 ar insemna 'legitimate'. La noi y=1 = phishing.
    out = merged[["url", "y"]]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\n[save] Merged dataset written to: {args.output}")
    print(f"[save] {len(out)} rows  phishing={int((out['y']==1).sum())}  legitimate={int((out['y']==0).sum())}")

    # Statistici lungime per clasa, pentru verificare
    print(f"\n=== Lungime URL (verificare bias) ===")
    for y_val, name in [(1, "Phishing"), (0, "Legitim")]:
        lens = out[out["y"] == y_val]["url"].str.len()
        print(f"  {name:<10}  mean={lens.mean():.1f}  median={lens.median():.0f}  min={lens.min()}  max={lens.max()}")


if __name__ == "__main__":
    main()