import asyncio
import pickle
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from playwright.async_api import async_playwright, TimeoutError as PwTimeout

BRANDS_DIR = Path("/app/screenshots/brands")
EMBEDDINGS_PATH = Path("/app/data/brand_embeddings.pkl")

# Crop height (px) used for embedding computation.
# Only the top portion of the screenshot is used — logo/header area is the most
# brand-distinctive region. Using the full page risks false positives because
# login forms look similar across sites.
CROP_HEIGHT = 300

BRANDS = [
    # ── Global — Tech ────────────────────────────────────────────────────────
    {"name": "google", "display": "Google", "category": "tech", "references": [
        {"label": "home",  "url": "https://www.google.com/"},
        {"label": "login", "url": "https://accounts.google.com/"},
    ]},
    {"name": "microsoft", "display": "Microsoft", "category": "tech", "references": [
        {"label": "home",  "url": "https://www.microsoft.com/ro-ro/"},
        {"label": "login", "url": "https://login.microsoftonline.com/"},
    ]},
    {"name": "apple", "display": "Apple", "category": "tech", "references": [
        {"label": "home",  "url": "https://www.apple.com/ro/"},
        {"label": "login", "url": "https://appleid.apple.com/"},
    ]},
    {"name": "github", "display": "GitHub", "category": "tech", "references": [
        {"label": "home",  "url": "https://github.com/"},
        {"label": "login", "url": "https://github.com/login"},
    ]},
    {"name": "dropbox", "display": "Dropbox", "category": "tech", "references": [
        {"label": "home",  "url": "https://www.dropbox.com/"},
        {"label": "login", "url": "https://www.dropbox.com/login"},
    ]},
    {"name": "adobe", "display": "Adobe", "category": "tech", "references": [
        {"label": "home",  "url": "https://www.adobe.com/"},
        {"label": "login", "url": "https://auth.services.adobe.com/ro_RO/index.html"},
    ]},
    {"name": "zoom", "display": "Zoom", "category": "tech", "references": [
        {"label": "home",  "url": "https://zoom.us/ro-ro/"},
        {"label": "login", "url": "https://zoom.us/signin"},
    ]},
    # ── Global — Social ──────────────────────────────────────────────────────
    {"name": "facebook", "display": "Facebook", "category": "social", "references": [
        {"label": "home",  "url": "https://www.facebook.com/"},
        {"label": "login", "url": "https://www.facebook.com/login"},
    ]},
    {"name": "instagram", "display": "Instagram", "category": "social", "references": [
        {"label": "home",  "url": "https://www.instagram.com/"},
        {"label": "login", "url": "https://www.instagram.com/accounts/login/"},
    ]},
    {"name": "linkedin", "display": "LinkedIn", "category": "social", "references": [
        {"label": "home",  "url": "https://www.linkedin.com/"},
        {"label": "login", "url": "https://www.linkedin.com/login"},
    ]},
    {"name": "twitter", "display": "Twitter / X", "category": "social", "references": [
        {"label": "login", "url": "https://x.com/login"},
    ]},
    {"name": "whatsapp", "display": "WhatsApp", "category": "social", "references": [
        {"label": "home", "url": "https://web.whatsapp.com/"},
    ]},
    # ── Global — Streaming / Gaming ──────────────────────────────────────────
    {"name": "netflix", "display": "Netflix", "category": "streaming", "references": [
        {"label": "home",  "url": "https://www.netflix.com/ro/"},
        {"label": "login", "url": "https://www.netflix.com/login"},
    ]},
    {"name": "spotify", "display": "Spotify", "category": "streaming", "references": [
        {"label": "home",  "url": "https://www.spotify.com/ro/"},
        {"label": "login", "url": "https://accounts.spotify.com/login"},
    ]},
    {"name": "youtube", "display": "YouTube", "category": "streaming", "references": [
        {"label": "home", "url": "https://www.youtube.com/"},
    ]},
    {"name": "steam", "display": "Steam", "category": "gaming", "references": [
        {"label": "home",  "url": "https://store.steampowered.com/"},
        {"label": "login", "url": "https://store.steampowered.com/login/"},
    ]},
    # ── Global — E-commerce / Travel ─────────────────────────────────────────
    {"name": "amazon", "display": "Amazon", "category": "ecommerce", "references": [
        {"label": "home",  "url": "https://www.amazon.com/"},
        {"label": "login", "url": "https://www.amazon.com/ap/signin"},
    ]},
    {"name": "ebay", "display": "eBay", "category": "ecommerce", "references": [
        {"label": "home",  "url": "https://www.ebay.com/"},
        {"label": "login", "url": "https://signin.ebay.com/signin/"},
    ]},
    {"name": "airbnb", "display": "Airbnb", "category": "travel", "references": [
        {"label": "home",  "url": "https://www.airbnb.com.ro/"},
        {"label": "login", "url": "https://www.airbnb.com.ro/login"},
    ]},
    {"name": "booking", "display": "Booking.com", "category": "travel", "references": [
        {"label": "home",  "url": "https://www.booking.com/"},
        {"label": "login", "url": "https://account.booking.com/sign-in"},
    ]},
    # ── Global — Finance ─────────────────────────────────────────────────────
    {"name": "paypal", "display": "PayPal", "category": "finance", "references": [
        {"label": "home",  "url": "https://www.paypal.com/"},
        {"label": "login", "url": "https://www.paypal.com/signin"},
    ]},
    {"name": "revolut", "display": "Revolut", "category": "finance", "references": [
        {"label": "home",  "url": "https://www.revolut.com/ro-RO/"},
        {"label": "login", "url": "https://sso.revolut.com/signin"},
    ]},
    {"name": "coinbase", "display": "Coinbase", "category": "finance", "references": [
        {"label": "home",  "url": "https://www.coinbase.com/"},
        {"label": "login", "url": "https://login.coinbase.com/"},
    ]},
    {"name": "binance", "display": "Binance", "category": "finance", "references": [
        {"label": "home",  "url": "https://www.binance.com/"},
        {"label": "login", "url": "https://accounts.binance.com/en/login"},
    ]},
    # ── Global — Logistics ───────────────────────────────────────────────────
    {"name": "dhl", "display": "DHL", "category": "logistics", "references": [
        {"label": "home", "url": "https://www.dhl.com/"},
    ]},
    # ── Romanian — Banking ───────────────────────────────────────────────────
    {"name": "bcr", "display": "BCR", "category": "banking-ro", "references": [
        {"label": "home",  "url": "https://www.bcr.ro/"},
        {"label": "login", "url": "https://login-business.bcr.ro/corporate-george-auth/login"},
    ]},
    {"name": "bancatransilvania", "display": "Banca Transilvania", "category": "banking-ro", "references": [
        {"label": "home",       "url": "https://www.bancatransilvania.ro/"},
        {"label": "george",     "url": "https://goapp.bancatransilvania.ro/app/auth/login"},
        {"label": "btpay",      "url": "https://btpay.bancatransilvania.ro/"},
        {"label": "btultra",    "url": "https://btultra.btrl.ro/btultraweb/_mcologon"},
    ]},
    {"name": "ing", "display": "ING România", "category": "banking-ro", "references": [
        {"label": "home", "url": "https://www.ing.ro/"},
    ]},
    {"name": "brd", "display": "BRD", "category": "banking-ro", "references": [
        {"label": "home", "url": "https://www.brd.ro/"},
    ]},
    {"name": "raiffeisen", "display": "Raiffeisen Bank", "category": "banking-ro", "references": [
        {"label": "home", "url": "https://www.raiffeisen.ro/"},
    ]},
    {"name": "cecbank", "display": "CEC Bank", "category": "banking-ro", "references": [
        {"label": "home", "url": "https://www.cec.ro/"},
    ]},
    # ── Romanian — E-commerce ────────────────────────────────────────────────
    {"name": "emag", "display": "eMAG", "category": "ecommerce-ro", "references": [
        {"label": "home",  "url": "https://www.emag.ro/"},
        {"label": "login", "url": "https://auth.emag.ro/user/login"},
    ]},
    {"name": "olx", "display": "OLX", "category": "ecommerce-ro", "references": [
        {"label": "home",  "url": "https://www.olx.ro/"},
        {"label": "login", "url": "https://login.olx.ro/"},
    ]},
    {"name": "altex", "display": "Altex", "category": "ecommerce-ro", "references": [
        {"label": "home", "url": "https://www.altex.ro/"},
    ]},
    {"name": "elefant", "display": "Elefant", "category": "ecommerce-ro", "references": [
        {"label": "home",  "url": "https://www.elefant.ro/"},
        {"label": "login", "url": "https://www.elefant.ro/login"},
    ]},
    {"name": "dedeman", "display": "Dedeman", "category": "ecommerce-ro", "references": [
        {"label": "home",  "url": "https://www.dedeman.ro/"},
        {"label": "login", "url": "https://www.dedeman.ro/ro/customer/account/login"},
    ]},
    {"name": "kaufland", "display": "Kaufland", "category": "retail-ro", "references": [
        {"label": "home", "url": "https://www.kaufland.ro/"},
    ]},
    # ── Romanian — Logistics ─────────────────────────────────────────────────
    {"name": "postaromana", "display": "Poșta Română", "category": "logistics-ro", "references": [
        {"label": "home",  "url": "https://www.posta-romana.ro/"},
        {"label": "login", "url": "https://www.posta-romana.ro/login.html"},
    ]},
    {"name": "fancourier", "display": "Fan Courier", "category": "logistics-ro", "references": [
        {"label": "home",  "url": "https://www.fancourier.ro/"},
        {"label": "login", "url": "https://www.selfawb.ro/new/login"},
    ]},
    {"name": "sameday", "display": "Sameday", "category": "logistics-ro", "references": [
        {"label": "home",  "url": "https://sameday.ro/"},
        {"label": "login", "url": "https://eawb.sameday.ro/login"},
    ]},
    {"name": "cargus", "display": "Cargus", "category": "logistics-ro", "references": [
        {"label": "home",  "url": "https://www.cargus.ro/"},
        {"label": "login", "url": "https://mycargus.cargus.ro/login"},
    ]},
    # ── Romanian — Gov ───────────────────────────────────────────────────────
    {"name": "anaf", "display": "ANAF", "category": "gov-ro", "references": [
        {"label": "home", "url": "https://www.anaf.ro/"},
    ]},
    {"name": "ghiseulro", "display": "Ghișeul.ro", "category": "gov-ro", "references": [
        {"label": "home",  "url": "https://www.ghiseul.ro/"},
        {"label": "login", "url": "https://www.ghiseul.ro/ghiseul/public/"},
    ]},
    {"name": "cnpp", "display": "CNPP", "category": "gov-ro", "references": [
        {"label": "home",  "url": "https://www.cnpp.ro/"},
        {"label": "login", "url": "https://www.cnpp.ro/autentificare"},
    ]},
    # ── Romanian — Delivery ──────────────────────────────────────────────────
    {"name": "tazz",  "display": "Tazz",  "category": "delivery-ro", "references": [
        {"label": "home", "url": "https://tazz.ro/"},
    ]},
    {"name": "glovo", "display": "Glovo", "category": "delivery-ro", "references": [
        {"label": "home",  "url": "https://glovoapp.com/ro/ro/"},
        {"label": "login", "url": "https://glovoapp.com/ro/login"},
    ]},
    {"name": "bolt", "display": "Bolt", "category": "delivery-ro", "references": [
        {"label": "home", "url": "https://bolt.eu/ro/"},
    ]},

    # ── Romanian — brands ──────────────────────────────────────────────────
    {"name": "zalando", "display": "Zalando", "category": "ecommerce-ro", "references": [
    {"label": "home",  "url": "https://www.zalando.ro/"},
    ]},
    {"name": "fashiondays", "display": "Fashion Days", "category": "ecommerce-ro", "references": [
        {"label": "home",  "url": "https://www.fashiondays.ro/"},
        {"label": "login", "url": "https://www.fashiondays.ro/customer/authentication"},
    ]},
    {"name": "aboutyou", "display": "About You", "category": "ecommerce-ro", "references": [
        {"label": "home",  "url": "https://www.aboutyou.ro/"},
        {"label": "login", "url": "https://www.aboutyou.ro/a/profile?loginFlow=register"},
    ]},
]


def load_clip_model():
    print("Loading CLIP ViT-B/32...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval()
    print("CLIP model ready\n")
    return model, preprocess


def compute_embedding(model, preprocess, image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    # Crop to top portion (logo/header area) to avoid false positives from
    # visually similar login form layouts across different sites.
    w, h = img.size
    img = img.crop((0, 0, w, min(CROP_HEIGHT, h)))
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().numpy()


async def capture(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"    [skip] screenshot exists")
        return True

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox",
                  "--disable-dev-shm-usage", "--disable-gpu"],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
            await _accept_cookies(page)
            await page.wait_for_timeout(1500)
            await page.screenshot(path=str(dest), full_page=False)
            return True
        except PwTimeout:
            print(f"    [FAIL] timeout")
            return False
        except Exception as e:
            print(f"    [FAIL] {e}")
            return False
        finally:
            await browser.close()


async def _accept_cookies(page) -> None:
    selectors = [
        "button:has-text('Accept all')", "button:has-text('Accept All')",
        "button:has-text('Accept cookies')", "button:has-text('I agree')",
        "button:has-text('Agree')", "button:has-text('OK')",
        "button:has-text('Got it')", "button:has-text('Allow all')",
        "button:has-text('Acceptă toate')", "button:has-text('Acceptă')",
        "button:has-text('Sunt de acord')", "button:has-text('De acord')",
    ]
    for sel in selectors:
        try:
            await page.locator(sel).first.click(timeout=1000)
            await page.wait_for_timeout(500)
            return
        except Exception:
            continue


async def main():
    BRANDS_DIR.mkdir(parents=True, exist_ok=True)
    Path("/app/data").mkdir(parents=True, exist_ok=True)

    embeddings: dict = {}
    if EMBEDDINGS_PATH.exists():
        with open(EMBEDDINGS_PATH, "rb") as f:
            embeddings = pickle.load(f)
        print(f"Resuming: {len(embeddings)} brands already processed\n")

    model, preprocess = load_clip_model()

    total = len(BRANDS)
    for i, brand in enumerate(BRANDS):
        name = brand["name"]
        refs = brand["references"]
        print(f"[{i+1:02d}/{total}] {brand['display']} ({len(refs)} reference(s))")

        existing_refs = embeddings.get(name, {}).get("references", [])
        existing_labels = {r["label"] for r in existing_refs}
        new_refs = list(existing_refs)
        changed = False

        for ref in refs:
            label = ref["label"]
            if label in existing_labels:
                print(f"  [{label}] skip")
                continue

            dest = BRANDS_DIR / f"{name}_{label}.png"
            ok = await capture(ref["url"], dest)
            if not ok:
                await asyncio.sleep(2)
                continue

            try:
                emb = compute_embedding(model, preprocess, str(dest))
                new_refs.append({
                    "label": label,
                    "url": ref["url"],
                    "screenshot": str(dest),
                    "embedding": emb,
                    "crop_height": CROP_HEIGHT,
                })
                print(f"  [{label}] OK — {emb.shape}")
                changed = True
            except Exception as e:
                print(f"  [{label}] embedding error: {e}")

            await asyncio.sleep(2)

        if changed or name not in embeddings:
            embeddings[name] = {
                "display": brand["display"],
                "category": brand["category"],
                "references": new_refs,
            }
            with open(EMBEDDINGS_PATH, "wb") as f:
                pickle.dump(embeddings, f)

    processed = sum(1 for v in embeddings.values() if v["references"])
    total_refs = sum(len(v["references"]) for v in embeddings.values())
    print(f"\nDone: {processed}/{total} brands, {total_refs} total embeddings")
    print(f"Saved to: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())