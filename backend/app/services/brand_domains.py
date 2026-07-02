from urllib.parse import urlparse

OFFICIAL_DOMAINS: dict[str, tuple[str, ...]] = {
    # Global — tech
    "google":      ("google.com",),
    "microsoft":   ("microsoft.com", "microsoftonline.com", "live.com", "office.com"),
    "apple":       ("apple.com", "icloud.com"),
    "github":      ("github.com",),
    "dropbox":     ("dropbox.com",),
    "adobe":       ("adobe.com",),
    "zoom":        ("zoom.us",),
    # Global — social
    "facebook":    ("facebook.com",),
    "instagram":   ("instagram.com",),
    "linkedin":    ("linkedin.com",),
    "twitter":     ("x.com", "twitter.com"),
    "whatsapp":    ("whatsapp.com",),
    # Global — streaming / gaming
    "netflix":     ("netflix.com",),
    "spotify":     ("spotify.com",),
    "youtube":     ("youtube.com",),
    "steam":       ("steampowered.com", "steamcommunity.com"),
    # Global — e-commerce / travel
    "amazon":      ("amazon.com",),
    "ebay":        ("ebay.com",),
    "airbnb":      ("airbnb.com", "airbnb.com.ro"),   # .com.ro = Romania
    "booking":     ("booking.com",),
    # Global — finance
    "paypal":      ("paypal.com",),
    "revolut":     ("revolut.com",),
    "coinbase":    ("coinbase.com",),
    "binance":     ("binance.com",),
    # Global — logistics
    "dhl":         ("dhl.com",),
    # Romanian — banking
    "bcr":              ("bcr.ro",),
    "bancatransilvania": ("bancatransilvania.ro", "btrl.ro"),
    "ing":              ("ing.ro",),
    "brd":              ("brd.ro",),
    "raiffeisen":       ("raiffeisen.ro",),
    "cecbank":          ("cec.ro",),
    "bnr":              ("bnr.ro"),
    # Romanian — e-commerce / retail
    "emag":        ("emag.ro",),
    "olx":         ("olx.ro",),
    "altex":       ("altex.ro",),
    "elefant":     ("elefant.ro",),
    "dedeman":     ("dedeman.ro",),
    "kaufland":    ("kaufland.ro",),
    "zalando":     ("zalando.ro",),
    "fashiondays": ("fashiondays.ro",),
    "aboutyou":    ("aboutyou.ro",),
    # Romanian — logistics
    "postaromana": ("posta-romana.ro",),
    "fancourier":  ("fancourier.ro", "selfawb.ro"),
    "sameday":     ("sameday.ro",),
    "cargus":      ("cargus.ro",),
    # Romanian — government
    "anaf":        ("anaf.ro",),
    "ghiseulro":   ("ghiseul.ro",),
    "cnpp":        ("cnpp.ro",),
    "roeid":       ("roeid.ro",),
    "cnas":       ("cnas.ro",),
    # Romanian — delivery
    "wolt":        ("wolt.ro",),
    "glovo":       ("glovoapp.com",),
    "bolt":        ("bolt.eu",),
}


def _extract_host(url: str) -> str:
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = (parsed.hostname or "").lower()
    except Exception:
        host = ""
    return host.removeprefix("www.")


def is_official_domain(url: str, brand: str) -> bool:
    host = _extract_host(url)
    if not host:
        return False
    for domain in OFFICIAL_DOMAINS.get(brand, ()):
        if host == domain or host.endswith("." + domain):
            return True
    return False