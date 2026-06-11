import json
import shutil
from pathlib import Path

DATASET_DIR = Path("/app/evaluation/phishpedia_dataset")
FILTERED_DIR = Path("/app/evaluation/phishpedia_filtered")

# Mapping: Phishpedia brand name -> cheia din baza noastra de embeddings
BRAND_MAPPING = {
    "Google":             "google",
    "Microsoft":          "microsoft",
    "Apple":              "apple",
    "GitHub":             "github",
    "Dropbox":            "dropbox",
    "Adobe Inc.":         "adobe",
    "Adobe":              "adobe",
    "Zoom":               "zoom",
    "Facebook":           "facebook",
    "Instagram":          "instagram",
    "LinkedIn":           "linkedin",
    "Twitter":            "twitter",
    "WhatsApp":           "whatsapp",
    "Netflix":            "netflix",
    "Spotify":            "spotify",
    "YouTube":            "youtube",
    "Steam":              "steam",
    "Amazon":             "amazon",
    "eBay":               "ebay",
    "Airbnb":             "airbnb",
    "Booking.com":        "booking",
    "PayPal":             "paypal",
    "Revolut":            "revolut",
    "Coinbase":           "coinbase",
    "Binance":            "binance",
    "DHL":                "dhl",
    "UPS":                "ups",
    "BCR":                "bcr",
    "Banca Transilvania": "bancatransilvania",
    "ING":                "ing",
    "BRD":                "brd",
    "Raiffeisen":         "raiffeisen",
    "CEC Bank":           "cecbank",
    "BNR":                "bnr",
    "eMAG":               "emag",
    "OLX":                "olx",
    "Altex":              "altex",
    "Elefant":            "elefant",
    "Dedeman":            "dedeman",
    "Kaufland":           "kaufland",
    "Posta Romana":       "postaromana",
    "Fan Courier":        "fancourier",
    "Sameday":            "sameday",
    "Cargus":             "cargus",
    "ANAF":               "anaf",
    "Ghiseul.ro":         "ghiseulro",
    "CNPP":               "cnpp",
    "ROeID":              "roeid",
    "CNSAS":              "cnsas",
    "Wolt":               "wolt",
    "Glovo":              "glovo",
    "Bolt":               "bolt",
    "Zalando":            "zalando",
    "Fashion Days":       "fashiondays",
    "About You":          "aboutyou",
}

def parse_info(info_path: Path) -> dict:
    try:
        text = info_path.read_text(encoding="utf-8").strip()
        # info.txt e un dict Python, nu JSON — folosim ast.literal_eval
        import ast
        return ast.literal_eval(text)
    except Exception:
        return {}

def main():
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)

    stats = {}
    skipped = 0
    copied = 0

    folders = sorted(DATASET_DIR.iterdir())
    total = len(folders)

    for i, folder in enumerate(folders):
        if not folder.is_dir():
            continue

        info_path = folder / "info.txt"
        shot_path = folder / "shot.png"

        if not info_path.exists() or not shot_path.exists():
            skipped += 1
            continue

        info = parse_info(info_path)
        brand_raw = info.get("brand", "")

        # Cauta in mapping
        brand_key = None
        for phish_name, our_key in BRAND_MAPPING.items():
            if phish_name.lower() in brand_raw.lower():
                brand_key = our_key
                break

        if brand_key is None:
            skipped += 1
            continue

        # Copiaza shot.png si info.txt in folderul filtrat
        dest = FILTERED_DIR / brand_key / folder.name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(shot_path, dest / "shot.png")
        shutil.copy2(info_path, dest / "info.txt")

        stats[brand_key] = stats.get(brand_key, 0) + 1
        copied += 1

        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{total}] copied={copied} skipped={skipped}")

    print(f"\nDone: {copied} samples copied, {skipped} skipped")
    print("\nPer-brand distribution:")
    for brand, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {brand:25s} {count:4d} samples")

if __name__ == "__main__":
    main()