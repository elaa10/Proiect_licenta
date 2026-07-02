
import sys

sys.path.insert(0, "/app")

from app.services.url_analyzer import extract_features, compute_lexical_score
from app.services.ml_classifier import predict_ml_score

CANDIDATE_URLS = [
    "https://www.example-shop.com/account/login",
    "http://my-online-store.net/cart/checkout",
    "https://news-portal-daily.com/article/123",
    "http://free-gift-cards.info/claim",
    "https://www.university-courses.org/login",
    "http://download-software-now.xyz/install",
    "https://job-application-form.com/apply",
    "http://customer-support-helpdesk.com/ticket",
    "https://www.local-restaurant-menu.com/order",
    "http://event-registration-page.net/signup",
    "https://www.fitness-tracker-app.com/dashboard",
    "http://survey-rewards-center.com/start",
    "https://www.travel-booking-site.com/reservation",
    "http://account-verification-needed.com/confirm",
    "https://www.online-banking-portal.net/login",
]


def classify(avg: float) -> str:
    if avg >= 0.55:
        return "phishing"
    if avg >= 0.30:
        return "suspicious"
    return "legitimate"


for url in CANDIDATE_URLS:
    features = extract_features(url)
    lexical = compute_lexical_score(features)
    ml = predict_ml_score(url)["score"]
    avg = (lexical + ml) / 2

    print(f"{url:<55} lexical={lexical:.3f}  ml={ml:.3f}  avg={avg:.3f}  -> {classify(avg)}")