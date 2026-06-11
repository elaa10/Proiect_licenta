from app.services.brand_domains import is_official_domain

BRAND_MATCH_THRESHOLD = 0.80

IMPERSONATION_THRESHOLD = 0.88


def _label(score: float) -> str:
    if score >= 0.55:
        return "phishing"
    if score >= 0.30:
        return "suspicious"
    return "legitimate"


def compute_verdict(
    url: str,
    lexical: float,
    ml: float | None,
    visual_brand: str | None,
    visual_similarity: float | None,
) -> dict:
    url_scores = [s for s in (lexical, ml) if s is not None]
    url_risk = sum(url_scores) / len(url_scores) if url_scores else 0.0

    brand_matched = bool(visual_brand) and bool(visual_similarity) \
        and visual_similarity >= BRAND_MATCH_THRESHOLD

    if not brand_matched:
        return {
            "verdict": _label(url_risk),
            "score": round(url_risk, 4),
            "reason": "url_only",
            "brand_impersonated": False,
        }

    if is_official_domain(url, visual_brand):
        return {
            "verdict": "legitimate",
            "score": round(min(url_risk, 0.25), 4),
            "reason": "official_domain_confirmed",
            "brand_impersonated": False,
        }

    if visual_similarity >= IMPERSONATION_THRESHOLD:
        score = max(url_risk, 0.90)
    else:
        score = max(url_risk, 0.50)

    return {
        "verdict": _label(score),
        "score": round(score, 4),
        "reason": "brand_impersonation",
        "brand_impersonated": True,
    }