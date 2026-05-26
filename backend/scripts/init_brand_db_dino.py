"""
Builds DINOv2 brand knowledge base.
Uses same screenshots as CLIP version — only embeddings differ.
Output: /app/data/brand_embeddings_dino.pkl
"""
import asyncio
import pickle
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from playwright.async_api import async_playwright, TimeoutError as PwTimeout

BRANDS_DIR = Path("/app/screenshots/brands")
EMBEDDINGS_PATH = Path("/app/data/brand_embeddings_dino.pkl")

CROP_STRATEGIES = [
    {"name": "top_150",  "top": 0,   "bottom": 150},
    {"name": "top_300",  "top": 0,   "bottom": 300},
    {"name": "top_500",  "top": 0,   "bottom": 500},
    {"name": "mid_300",  "top": 100, "bottom": 400},
]

COOKIE_TEXTS = [
    'Permite toate modulele cookie', 'Allow all cookies',
    'Accept all', 'Accept All', 'Acceptă toate',
    'I agree', 'Agree', 'Allow all', 'Got it', 'OK',
    'De acord', 'Sunt de acord', 'I accept', 'Accept cookies',
    'Continuă', 'Continue', 'I understand', 'Am înțeles',
]

COOKIE_SELECTORS = [
    "#onetrust-accept-btn-handler",
    "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
    "#truste-consent-button",
    "div[role='button']:has-text('Permite toate modulele cookie')",
    "div[role='button']:has-text('Allow all cookies')",
    "div[role='button']:has-text('Accept all')",
    "button:has-text('Accept all')", "button:has-text('Accept All')",
    "button:has-text('Accept cookies')", "button:has-text('I agree')",
    "button:has-text('Agree')", "button:has-text('Allow all')",
    "button:has-text('OK')", "button:has-text('Got it')",
    "button:has-text('Acceptă toate')", "button:has-text('Acceptă')",
    "button:has-text('Sunt de acord')", "button:has-text('De acord')",
    "button:has-text('Continuă')", "button:has-text('Am înțeles')",
    "#accept-cookies", "#cookie-accept", ".accept-cookies",
    "[aria-label='Accept cookies']", "[data-gdpr-accept='all']",
]

BRANDS = [
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
    {"name": "dhl", "display": "DHL", "category": "logistics", "references": [
        {"label": "home", "url": "https://www.dhl.com/"},
    ]},
    {"name": "bcr", "display": "BCR", "category": "banking-ro", "references": [
        {"label": "home",  "url": "https://www.bcr.ro/"},
        {"label": "login", "url": "https://login-business.bcr.ro/corporate-george-auth/login"},
    ]},
    {"name": "bancatransilvania", "display": "Banca Transilvania", "category": "banking-ro", "references": [
        {"label": "home",    "url": "https://www.bancatransilvania.ro/"},
        {"label": "george",  "url": "https://goapp.bancatransilvania.ro/app/auth/login"},
        {"label": "btpay",   "url": "https://btpay.bancatransilvania.ro/"},
        {"label": "btultra", "url": "https://btultra.btrl.ro/btultraweb/_mcologon"},
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
    {"name": "zalando", "display": "Zalando", "category": "ecommerce-ro", "references": [
        {"label": "home", "url": "https://www.zalando.ro/"},
    ]},
    {"name": "fashiondays", "display": "Fashion Days", "category": "ecommerce-ro", "references": [
        {"label": "home",  "url": "https://www.fashiondays.ro/"},
        {"label": "login", "url": "https://www.fashiondays.ro/customer/authentication"},
    ]},
    {"name": "aboutyou", "display": "About You", "category": "ecommerce-ro", "references": [
        {"label": "home",  "url": "https://www.aboutyou.ro/"},
        {"label": "login", "url": "https://www.aboutyou.ro/a/profile?loginFlow=register"},
    ]},
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
    {"name": "tazz", "display": "Tazz", "category": "delivery-ro", "references": [
        {"label": "home", "url": "https://tazz.ro/"},
    ]},
    {"name": "glovo", "display": "Glovo", "category": "delivery-ro", "references": [
        {"label": "home",  "url": "https://glovoapp.com/ro/ro/"},
        {"label": "login", "url": "https://glovoapp.com/ro/login"},
    ]},
    {"name": "bolt", "display": "Bolt", "category": "delivery-ro", "references": [
        {"label": "home", "url": "https://bolt.eu/ro/"},
    ]},
]


def load_dino_model():
    print("Loading DINOv2 (facebook/dinov2-base)...")
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model.eval()
    print("DINOv2 model ready\n")
    return model, processor


def compute_multi_crop_embeddings(model, processor, image_path: str) -> list:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"    [FAIL] image load: {e}")
        return []

    w, h = img.size
    embeddings = []
    for strategy in CROP_STRATEGIES:
        top = strategy["top"]
        bottom = min(strategy["bottom"], h)
        if bottom <= top:
            continue
        crop = img.crop((0, top, w, bottom))
        try:
            inputs = processor(images=crop, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :]
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.squeeze().numpy())
        except Exception:
            continue
    return embeddings


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
        try:
            parsed = urlparse(url)
            parts = (parsed.hostname or "").split(".")
            root = "." + ".".join(parts[-2:]) if len(parts) >= 2 else parsed.hostname
            await context.add_cookies([
                {"name": "OptanonAlertBoxClosed", "value": "true", "domain": root, "path": "/"},
                {"name": "cookieconsent_status", "value": "dismiss", "domain": root, "path": "/"},
                {"name": "cookies_accepted", "value": "true", "domain": root, "path": "/"},
            ])
        except Exception:
            pass

        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
            await page.wait_for_timeout(2000)
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
    for sel in COOKIE_SELECTORS:
        try:
            locator = page.locator(sel).first
            if await locator.is_visible(timeout=600):
                await locator.click(timeout=600)
                await page.wait_for_timeout(500)
                return
        except Exception:
            continue
    try:
        await page.evaluate(f"""
            () => {{
                const texts = {COOKIE_TEXTS};
                const els = [...document.querySelectorAll('div[role="button"]'),
                             ...document.querySelectorAll('button')];
                for (const el of els) {{
                    if (texts.some(t => el.textContent.trim().includes(t))) {{
                        el.click(); return true;
                    }}
                }}
            }}
        """)
        await page.wait_for_timeout(500)
    except Exception:
        pass


async def main():
    BRANDS_DIR.mkdir(parents=True, exist_ok=True)
    Path("/app/data").mkdir(parents=True, exist_ok=True)

    embeddings: dict = {}
    if EMBEDDINGS_PATH.exists():
        with open(EMBEDDINGS_PATH, "rb") as f:
            embeddings = pickle.load(f)
        print(f"Resuming: {len(embeddings)} brands already processed\n")

    model, processor = load_dino_model()
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

            # Reuse existing screenshot from CLIP version
            dest = BRANDS_DIR / f"{name}_{label}.png"
            ok = await capture(ref["url"], dest)
            if not ok:
                await asyncio.sleep(2)
                continue

            embs = compute_multi_crop_embeddings(model, processor, str(dest))
            if embs:
                new_refs.append({
                    "label": label,
                    "url": ref["url"],
                    "screenshot": str(dest),
                    "embeddings": embs,
                })
                print(f"  [{label}] OK — {len(embs)} crop embeddings")
                changed = True
            else:
                print(f"  [{label}] embedding error")

            await asyncio.sleep(1)

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
    print(f"\nDone: {processed}/{total} brands, {total_refs} total references")
    print(f"Saved to: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())