import re
from urllib.parse import urlparse

SUSPICIOUS_KEYWORDS = [
    "login", "secure", "verify", "update", "account", "banking",
    "confirm", "password", "credential", "alert", "suspended",
    "unusual", "activity", "signin", "webscr", "cmd", "ebayisapi",
]

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "buff.ly",
    "short.link", "rb.gy", "cutt.ly", "is.gd", "v.gd", "tiny.cc",
}

SUSPICIOUS_TLDS = {
    ".top", ".cyou", ".xin", ".gq", ".ml", ".cf", ".tk", ".pw",
    ".buzz", ".club", ".icu", ".live", ".xyz", ".work", ".rest",
}

TOP_BRANDS = [
    "paypal", "google", "apple", "microsoft", "amazon", "facebook",
    "instagram", "netflix", "steam", "ebay", "bcr", "ing", "banca",
    "raiffeisen", "emag", "revolut", "dropbox", "linkedin", "twitter",
]

_IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def extract_features(url: str) -> dict:
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
    except Exception:
        return _default_features()

    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    full = url.lower()

    domain = hostname.removeprefix("www.")
    parts = domain.split(".")
    num_subdomains = max(0, len(parts) - 2)
    tld = "." + parts[-1] if parts else ""
    sld = parts[-2] if len(parts) >= 2 else domain

    # Also check the first token before a hyphen — catches "paypa1-secure", "emag-premiu"
    sld_first_token = sld.split("-")[0]
    min_lev = min(
        min((_levenshtein(sld, b) for b in TOP_BRANDS), default=99),
        min((_levenshtein(sld_first_token, b) for b in TOP_BRANDS), default=99),
    )

    # Total slash count, excluding the "://" of the scheme
    slash_count = url.count("/") - (2 if "://" in url else 0)

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": hostname.count("-"),
        "num_slashes": max(slash_count, 0),
        "num_underscores": url.count("_"),
        "num_question_marks": url.count("?"),
        "has_at_symbol": int("@" in url),
        "num_subdomains": num_subdomains,
        "has_ip_address": int(bool(_IP_RE.match(hostname))),
        "is_https": int(parsed.scheme == "https"),
        "is_url_shortener": int(hostname in URL_SHORTENERS),
        "is_punycode": int("xn--" in hostname),
        "suspicious_keyword_count": sum(kw in full for kw in SUSPICIOUS_KEYWORDS),
        "digit_ratio": sum(c.isdigit() for c in hostname) / max(len(hostname), 1),
        "has_suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "double_slash_in_path": int("//" in path),
        "min_brand_levenshtein": min_lev,
    }


def compute_lexical_score(features: dict) -> float:
    score = 0.0

    if features["url_length"] > 75:
        score += 0.15
    elif features["url_length"] > 54:
        score += 0.05

    if features["has_ip_address"]:
        score += 0.20

    if features["has_at_symbol"]:
        score += 0.20

    if features["is_punycode"]:
        score += 0.25

    if features["is_url_shortener"]:
        score += 0.15

    if features["has_suspicious_tld"]:
        score += 0.10

    score += min(features["suspicious_keyword_count"] * 0.05, 0.20)

    if features["num_subdomains"] > 3:
        score += 0.10
    elif features["num_subdomains"] > 1:
        score += 0.05

    if features["num_slashes"] > 5:
        score += 0.10
    elif features["num_slashes"] > 3:
        score += 0.05

    if features["num_underscores"] > 2:
        score += 0.05

    if features["num_question_marks"] > 1:
        score += 0.05

    if features["digit_ratio"] > 0.3:
        score += 0.05

    if features["double_slash_in_path"]:
        score += 0.05

    lev = features["min_brand_levenshtein"]
    if 1 <= lev <= 3:
        score += 0.15

    if not features["is_https"]:
        score += 0.03

    return round(min(score, 1.0), 4)


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[-1] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _default_features() -> dict:
    return {k: 0 for k in [
        "url_length", "hostname_length", "path_length", "num_dots",
        "num_hyphens", "num_slashes", "num_underscores", "num_question_marks",
        "has_at_symbol", "num_subdomains", "has_ip_address", "is_https",
        "is_url_shortener", "is_punycode", "suspicious_keyword_count",
        "digit_ratio", "has_suspicious_tld", "double_slash_in_path",
        "min_brand_levenshtein",
    ]}