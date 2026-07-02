

import argparse
import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, "/app")
from app.services.url_analyzer import extract_features

DEFAULT_MODEL = "/app/models/rf_model.joblib"

FEATURE_ORDER = [
    "url_length", "hostname_length", "path_length",
    "num_dots", "num_hyphens", "num_slashes", "num_underscores", "num_question_marks",
    "has_at_symbol", "num_subdomains", "has_ip_address", "is_https",
    "is_url_shortener", "is_punycode", "suspicious_keyword_count",
    "digit_ratio", "has_suspicious_tld", "double_slash_in_path",
    "min_brand_levenshtein", "sld_is_exact_brand",
    "has_confusable_chars",
]

TEST_URLS = [
    # ===== LEGITIME — brand-uri internationale =====
    ("https://google.com",                              "LEGIT", "Brand fara www"),
    ("https://www.google.com",                          "LEGIT", "Brand cu www"),
    ("https://www.google.com/search?q=phishing",        "LEGIT", "Brand cu path + query"),
    ("https://paypal.com",                              "LEGIT", "Brand financiar"),
    ("https://github.com/torvalds/linux",               "LEGIT", "Path lung cu nume reale"),
    ("https://stackoverflow.com/questions/tagged/python", "LEGIT", "Forum tehnic"),
    ("https://en.wikipedia.org/wiki/Phishing",          "LEGIT", "Wikipedia"),
    ("https://www.bbc.com/news/world-europe",           "LEGIT", "Site stiri"),

    # ===== LEGITIME — brand-uri romanesti (NU sunt in dataset, test de generalizare) =====
    ("https://bcr.ro",                                  "LEGIT", "Banca RO scurt"),
    ("https://www.bcr.ro/ro/persoane-fizice",           "LEGIT", "Banca RO cu path"),
    ("https://www.emag.ro",                             "LEGIT", "E-commerce RO"),
    ("https://www.anaf.ro/anaf/internet",               "LEGIT", "Institutie publica RO"),
    ("https://ghiseul.ro",                              "LEGIT", "Plati taxe RO"),

    # ===== PHISHING — typosquatting =====
    ("http://paypa1-secure.com/login",                  "PHISH", "Typosquat (paypal -> paypa1)"),
    ("http://g00gle-account.com/verify",                "PHISH", "Typosquat (google -> g00gle)"),
    ("http://amaz0n-prime.com/order",                   "PHISH", "Typosquat (amazon -> amaz0n)"),

    # ===== PHISHING — brand pe TLD suspect =====
    ("http://google.top/account",                       "PHISH", "Brand exact pe TLD suspect"),
    ("https://service-paypal.xyz/signin",               "PHISH", "Brand + TLD suspect"),
    ("http://emag-premiu.cyou/castigator",              "PHISH", "Smishing RO clasic"),
    ("http://bcr-secure.tk/verify-account",             "PHISH", "Brand RO + TLD suspect"),

    # ===== PHISHING — alte semnale =====
    ("http://192.168.1.1/login.php",                    "PHISH", "Adresa IP in loc de domeniu"),
    ("https://account-update-microsoft.tk/login.html",  "PHISH", "Subdomenii + TLD suspect"),
    ("https://xn--80ak6aa92e.com/signin",               "PHISH", "Punycode (IDN homograph)"),
    ("http://anaf-rambursare.gq/formular",              "PHISH", "Institutie RO impersonata"),
]


def evaluate(model, urls, threshold: float = 0.5):
    results = []
    correct = 0
    for url, expected, scenario in urls:
        f = extract_features(url)
        x = np.array([[f[k] for k in FEATURE_ORDER]], dtype=np.float32)
        proba = float(model.predict_proba(x)[0][1])
        actual = "PHISH" if proba >= threshold else "LEGIT"
        ok = actual == expected
        if ok:
            correct += 1
        results.append({
            "url": url,
            "expected": expected,
            "score": round(proba, 4),
            "verdict": actual,
            "correct": ok,
            "scenario": scenario,
        })
    return results, correct


def print_report(results, correct, total, threshold):
    print(f"\n{'=' * 100}")
    print(f"SANITY CHECK — Random Forest model  (threshold τ={threshold})")
    print(f"{'=' * 100}\n")

    legit_results = [r for r in results if r["expected"] == "LEGIT"]
    legit_correct = sum(1 for r in legit_results if r["correct"])
    print(f"--- URL-uri LEGITIME ({legit_correct}/{len(legit_results)} corecte) ---")
    print(f"{'URL':<55} {'Score':>8}  Verdict  Scenario")
    print("-" * 100)
    for r in legit_results:
        mark = "✓" if r["correct"] else "✗"
        print(f"{r['url']:<55} {r['score']:>8.4f}  {r['verdict']:<7}  {mark} {r['scenario']}")

    phish_results = [r for r in results if r["expected"] == "PHISH"]
    phish_correct = sum(1 for r in phish_results if r["correct"])
    print(f"\n--- URL-uri PHISHING ({phish_correct}/{len(phish_results)} corecte) ---")
    print(f"{'URL':<55} {'Score':>8}  Verdict  Scenario")
    print("-" * 100)
    for r in phish_results:
        mark = "✓" if r["correct"] else "✗"
        print(f"{r['url']:<55} {r['score']:>8.4f}  {r['verdict']:<7}  {mark} {r['scenario']}")

    print(f"\n{'=' * 100}")
    print(f"TOTAL: {correct}/{total} corecte ({100*correct/total:.1f}%)")
    print(f"  Legitime: {legit_correct}/{len(legit_results)} ({100*legit_correct/len(legit_results):.1f}%) — precision pentru clasa LEGIT")
    print(f"  Phishing: {phish_correct}/{len(phish_results)} ({100*phish_correct/len(phish_results):.1f}%) — recall pentru clasa PHISH")
    print(f"{'=' * 100}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None, help="Optional: scrie rezultatele in JSON")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        print("Mai intai antreneaza modelul: python scripts/train_rf.py")
        sys.exit(1)

    print(f"[load] Loading model from {args.model} ...")
    data = joblib.load(args.model)
    model = data["model"]
    print(f"[load] Model trained at: {data.get('trained_at', 'N/A')}")

    results, correct = evaluate(model, TEST_URLS, threshold=args.threshold)
    print_report(results, correct, len(TEST_URLS), args.threshold)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "model_path": args.model,
                "threshold": args.threshold,
                "total": len(TEST_URLS),
                "correct": correct,
                "accuracy": round(correct / len(TEST_URLS), 4),
                "results": results,
            }, f, indent=2)
        print(f"[save] Results: {args.output}")


if __name__ == "__main__":
    main()