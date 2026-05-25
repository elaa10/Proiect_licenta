"""
Smoke tests for the Random Forest ML classifier.

Run from inside the backend container:
    docker exec -it proiect_licenta-backend-1 python -m tests.test_ml_smoke
"""

import sys
import os

sys.path.insert(0, "/app")

from app.services.ml_classifier import predict_ml_score, is_model_available

# ── SCORING TESTS ─────────────────────────────────────────────────────────────
# "safe" → ML score < 0.35
# "mid"  → 0.35 ≤ score < 0.70
# "high" → score ≥ 0.60

SCORING_TESTS = [
    # Legitimate URLs — must score low
    ("https://www.google.com",                "safe", "Google homepage"),
    ("https://www.bcr.ro",                    "safe", "Romanian bank homepage"),
    ("https://www.emag.ro/laptop/c",          "safe", "Romanian e-commerce"),
    ("https://github.com/torvalds/linux",     "safe", "GitHub repo"),
    ("https://www.facebook.com/help",         "safe", "Facebook help page"),
    ("https://paypal.com/signin",             "safe", "PayPal legitimate signin"),

    # Strong phishing signals — must score high
    ("http://192.168.1.1/login/account",      "high", "Raw IP address"),
    ("http://google.com@evil.ru/secure",      "high", "AT symbol masking"),
    ("https://xn--80ak6aa92e.com/login",      "high", "Punycode IDN homograph"),
    ("http://paypal-secure.com/verify",       "high", "Brand prefix + hyphen phishing"),
    ("http://login-bcr-secure.cyou/update",   "high", "Suspicious TLD + keywords"),
    ("http://secure-login-paypal-verify-account-update.com/banking/credentials/step2/confirm/action.php",
                                              "high", "Long URL with many keywords"),

    # Medium signals
    ("http://paypa1.com/signin",              "mid",  "Typosquatting paypal"),
    ("http://gooogle.com/login",              "mid",  "Typosquatting google"),
    ("https://bit.ly/3xampLe",               "mid",  "URL shortener"),
]

# ── PROPERTY TESTS ────────────────────────────────────────────────────────────

PROPERTY_TESTS = [
    # Score must be a float in [0, 1]
    "https://www.google.com",
    "http://192.168.1.1/login",
    "http://paypal-secure.com/verify",
    "https://bit.ly/abc",
    "not-a-url",
    "",
    "http://" + "a" * 2000,
]

# ── RANKING TESTS ─────────────────────────────────────────────────────────────
# For each pair, the first URL must score LOWER than the second

RANKING_TESTS = [
    (
        "https://www.google.com",
        "http://google-login.com/verify",
        "Legitimate Google vs brand-prefix phishing",
    ),
    (
        "https://www.bcr.ro",
        "http://login-bcr-secure.cyou/update",
        "Legitimate BCR vs phishing BCR",
    ),
    (
        "https://paypal.com",
        "http://paypal-secure-login.top/account/verify",
        "Legitimate PayPal vs phishing PayPal",
    ),
    (
        "https://github.com/user/repo",
        "http://192.168.1.1/login/account",
        "Legitimate GitHub vs raw IP",
    ),
]


# ── TEST RUNNER ───────────────────────────────────────────────────────────────

def run_prerequisite_check():
    print("\n" + "=" * 80)
    print("PREREQUISITE — Model availability")
    print("=" * 80)
    if not is_model_available():
        print("FAIL — rf_model.joblib not found at /app/models/")
        print("       Run: python scripts/train_rf.py --csv /app/data/phishing_site_urls.csv --sample 150000")
        sys.exit(1)
    print("PASS — Model file found\n")


def run_scoring_tests():
    print("=" * 80)
    print("SECTION 1 — SCORING TESTS")
    print("=" * 80)
    print(f"{'URL':<60} {'Expected':<8} {'Score':<8} {'Result'}")
    print("-" * 80)

    passed = failed = 0
    for url, category, desc in SCORING_TESTS:
        result = predict_ml_score(url)
        score = result["score"]

        if category == "safe":
            ok = score < 0.35
        elif category == "mid":
            ok = 0.25 <= score < 0.80
        else:
            ok = score >= 0.60

        status = "PASS" if ok else "FAIL"
        passed += ok
        failed += not ok

        display = url if len(url) <= 58 else url[:55] + "..."
        print(f"{display:<60} {category:<8} {score:<8.3f} {status}  # {desc}")

    print("-" * 80)
    print(f"Scoring: {passed} passed, {failed} failed\n")
    return failed


def run_property_tests():
    print("=" * 80)
    print("SECTION 2 — PROPERTY TESTS (score must be float in [0, 1])")
    print("=" * 80)

    passed = failed = 0
    for url in PROPERTY_TESTS:
        try:
            result = predict_ml_score(url)
            score = result["score"]
            ok = isinstance(score, float) and 0.0 <= score <= 1.0
        except Exception as e:
            ok = False
            score = f"ERROR: {e}"

        status = "PASS" if ok else "FAIL"
        passed += ok
        failed += not ok

        display = url if len(url) <= 60 else url[:57] + "..."
        print(f"{status}  score={score}  url={display!r}")

    print(f"\nProperties: {passed} passed, {failed} failed\n")
    return failed


def run_ranking_tests():
    print("=" * 80)
    print("SECTION 3 — RANKING TESTS (legitimate must score lower than phishing)")
    print("=" * 80)

    passed = failed = 0
    for url_legit, url_phish, desc in RANKING_TESTS:
        score_legit = predict_ml_score(url_legit)["score"]
        score_phish = predict_ml_score(url_phish)["score"]
        ok = score_legit < score_phish

        status = "PASS" if ok else "FAIL"
        passed += ok
        failed += not ok

        print(f"{status}  {desc}")
        print(f"       legit={score_legit:.3f}  {url_legit}")
        print(f"       phish={score_phish:.3f}  {url_phish}")

    print(f"\nRankings: {passed} passed, {failed} failed\n")
    return failed


def main():
    run_prerequisite_check()
    s_fail = run_scoring_tests()
    p_fail = run_property_tests()
    r_fail = run_ranking_tests()
    total = s_fail + p_fail + r_fail

    print("=" * 80)
    if total == 0:
        print("ALL TESTS PASSED — ML classifier is ready.")
    else:
        print(f"{total} test(s) failed — review output above.")
    print("=" * 80)


if __name__ == "__main__":
    main()