"""
Smoke tests for the lexical URL detector.

Run from inside the backend container:
    docker exec -it proiect_licenta-backend-1 python -m tests.test_lexical_smoke
"""

from app.services.url_analyzer import extract_features, compute_lexical_score


# ── SCORING TESTS ─────────────────────────────────────────────────────────────
# "safe" → score < 0.20
# "mid"  → 0.20 ≤ score < 0.65
# "high" → score ≥ 0.40

SCORING_TESTS = [
    # Legitimate URLs — must score below the safe threshold
    ("https://www.google.com",               "safe", "Google homepage"),
    ("https://github.com/torvalds/linux",    "safe", "GitHub repo"),
    ("https://www.bcr.ro",                   "safe", "Romanian bank"),
    ("https://www.emag.ro/produs/iphone-15", "safe", "Romanian e-commerce"),
    ("https://www.facebook.com/help",        "safe", "Facebook help page"),
    # Deep legitimate path — SLD matches brand exactly (lev=0), score stays low
    ("http://paypal.com/a/b/c/d/e/f/g/h",   "safe", "Legitimate domain with deep path"),

    # Strong phishing signals — must reach the high threshold
    ("http://192.168.1.1/login/account",     "high", "IP address in host"),
    ("http://google.com@evil.ru/secure",     "high", "AT symbol masking"),
    ("https://xn--80ak6aa92e.com/login",     "high", "Punycode / IDN homograph"),
    ("https://xn--pple-43d.com/signin",      "high", "Punycode: apple homograph"),
    ("http://secure-login-paypal-verify-account-update.com/banking/credentials/step2/confirm/action.php",
                                             "high", "Long URL with many keywords"),

    # Medium signals — must fall in the 0.20–0.65 range.
    # pre-hyphen token "paypa1" has Levenshtein distance 1 from "paypal"
    ("http://paypa1-secure.com/verify",      "mid",  "Typosquatting with suffix"),
    ("http://gooogle.com/login",             "mid",  "One-char typosquatting"),
    ("http://login-bcr-secure.cyou/update",  "mid",  "Suspicious TLD + keywords"),
    # URL shortener; short SLD "bit" coincidentally has lev-distance 2 from "bcr"
    ("https://bit.ly/3xampLe",              "mid",  "URL shortener"),
    ("http://ing-romania.secure-banking-login.verify.top/account/update?id=1234",
                                             "mid",  "Subdomains + keywords + suspicious TLD"),
    ("http://ing_ro_secure_login_bank.top/", "mid",  "Underscores + suspicious TLD + keywords"),
    # "evil" has lev-distance 3 from "emag"; combined with no-HTTPS and multiple
    # question marks the heuristic reaches the mid range
    ("http://evil.com/x?a=1&b=2?c=3?d=4",  "mid",  "Multiple question marks"),
]


# ── FEATURE UNIT TESTS ────────────────────────────────────────────────────────

FEATURE_TESTS = [
    # url_length (https://www.google.com = 22 chars)
    ("https://www.google.com", "url_length", 22, "URL length exact"),

    # has_ip_address
    ("http://192.168.1.1/login", "has_ip_address", 1, "IP detected"),
    ("https://www.google.com",   "has_ip_address", 0, "No IP"),

    # has_at_symbol
    ("http://user@evil.com/path", "has_at_symbol", 1, "AT symbol present"),
    ("https://www.google.com",    "has_at_symbol", 0, "No AT symbol"),

    # is_https
    ("https://example.com", "is_https", 1, "HTTPS detected"),
    ("http://example.com",  "is_https", 0, "HTTP detected"),

    # is_url_shortener
    ("https://bit.ly/abc123",   "is_url_shortener", 1, "bit.ly shortener"),
    ("https://tinyurl.com/xyz", "is_url_shortener", 1, "tinyurl shortener"),
    ("https://www.google.com",  "is_url_shortener", 0, "Not a shortener"),

    # is_punycode
    ("https://xn--80ak6aa92e.com/login", "is_punycode", 1, "Punycode domain detected"),
    ("https://xn--pple-43d.com",         "is_punycode", 1, "Another Punycode domain"),
    ("https://www.apple.com",            "is_punycode", 0, "Normal ASCII domain"),

    # num_slashes
    ("http://evil.com/a/b/c/d/e", "num_slashes", 5, "5 slashes in path"),
    ("https://google.com",         "num_slashes", 0, "No path slashes"),

    # num_underscores
    ("http://secure_login_bank.com/account_update", "num_underscores", 3, "3 underscores"),
    ("https://www.google.com",                       "num_underscores", 0, "No underscores"),

    # num_question_marks
    ("http://evil.com/x?a=1?b=2?c=3", "num_question_marks", 3, "3 question marks"),
    ("https://google.com/search?q=hi", "num_question_marks", 1, "1 question mark"),
    ("https://www.google.com",         "num_question_marks", 0, "No question marks"),

    # has_suspicious_tld
    ("http://login.top",  "has_suspicious_tld", 1, ".top is suspicious"),
    ("http://login.cyou", "has_suspicious_tld", 1, ".cyou is suspicious"),
    ("http://login.com",  "has_suspicious_tld", 0, ".com is not suspicious"),
    ("https://login.ro",  "has_suspicious_tld", 0, ".ro is not suspicious"),

    # suspicious_keyword_count ("bank" is NOT in the list; only "banking" is)
    ("http://secure-login-verify.com",  "suspicious_keyword_count", 3, "secure + login + verify"),
    ("http://ing-secure-login-bank.com","suspicious_keyword_count", 2, "secure + login (bank != banking)"),
    ("https://www.google.com",          "suspicious_keyword_count", 0, "No keywords"),

    # double_slash_in_path
    ("http://evil.com//redirect/login", "double_slash_in_path", 1, "Double slash in path"),
    ("https://www.google.com/search",   "double_slash_in_path", 0, "No double slash"),

    # num_subdomains (a.b.c.evil.com → 3 subdomains: a, b, c)
    ("http://a.b.c.evil.com/", "num_subdomains", 3, "3 subdomains: a, b, c"),
    ("https://www.google.com",  "num_subdomains", 0, "www stripped → 0 subdomains"),

    # min_brand_levenshtein — whole SLD
    ("http://paypa1.com",  "min_brand_levenshtein", 1, "paypa1 → paypal, distance 1"),
    ("http://gooogle.com", "min_brand_levenshtein", 1, "gooogle → google, distance 1"),
    ("http://paypal.com",  "min_brand_levenshtein", 0, "paypal → paypal, distance 0"),

    # min_brand_levenshtein — pre-hyphen token (key fix: compound phishing domains)
    ("http://paypa1-secure.com", "min_brand_levenshtein", 1, "pre-hyphen paypa1 → paypal = 1"),

    # sld_is_exact_brand
    ("https://paypal.com",        "sld_is_exact_brand", 1, "paypal is in TOP_BRANDS"),
    ("https://google.com",        "sld_is_exact_brand", 1, "google is in TOP_BRANDS"),
    ("https://paypal-secure.com", "sld_is_exact_brand", 0, "paypal-secure is not an exact match"),
    ("https://random-site.com",   "sld_is_exact_brand", 0, "unknown SLD → 0"),
]


# ── TEST RUNNER ───────────────────────────────────────────────────────────────

def run_scoring_tests():
    print("\n" + "="*90)
    print("SECTION 1 — SCORING TESTS")
    print("="*90)
    print(f"{'URL':<65} {'Expected':<8} {'Score':<8} {'Result'}")
    print("-"*90)

    passed = failed = 0
    for url, category, desc in SCORING_TESTS:
        score = compute_lexical_score(extract_features(url))
        if category == "safe":
            ok = score < 0.20
        elif category == "mid":
            ok = 0.20 <= score < 0.65
        else:
            ok = score >= 0.40

        status = "PASS" if ok else "FAIL"
        passed += ok
        failed += not ok

        display = url if len(url) <= 63 else url[:60] + "..."
        print(f"{display:<65} {category:<8} {score:<8.3f} {status}  # {desc}")

    print("-"*90)
    print(f"Scoring: {passed} passed, {failed} failed\n")
    return failed


def run_feature_tests():
    print("="*90)
    print("SECTION 2 — FEATURE UNIT TESTS")
    print("="*90)
    print(f"{'Feature':<28} {'URL':<43} {'Expected':<12} {'Got':<12} {'Result'}")
    print("-"*90)

    passed = failed = 0
    for url, feature, expected, desc in FEATURE_TESTS:
        features = extract_features(url)
        got = features.get(feature)
        ok = (got == expected)
        status = "PASS" if ok else "FAIL"
        passed += ok
        failed += not ok

        url_display = url if len(url) <= 41 else url[:38] + "..."
        print(f"{feature:<28} {url_display:<43} {str(expected):<12} {str(got):<12} {status}  # {desc}")

    print("-"*90)
    print(f"Features: {passed} passed, {failed} failed\n")
    return failed


def main():
    s_fail = run_scoring_tests()
    f_fail = run_feature_tests()
    total = s_fail + f_fail

    print("="*90)
    if total == 0:
        print("ALL TESTS PASSED — lexical detector is ready for production.")
    else:
        print(f"{total} test(s) failed — review output above.")
    print("="*90)


if __name__ == "__main__":
    main()