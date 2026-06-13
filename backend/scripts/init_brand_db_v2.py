import asyncio
import pickle
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional

import numpy as np
import open_clip
import torch
from PIL import Image, ImageChops
from playwright.async_api import async_playwright, TimeoutError as PwTimeout

BRANDS_DIR = Path("/app/screenshots/brands")
EMBEDDINGS_PATH = Path("/app/data/brand_embeddings_v2.pkl")

# Strategia hibridă (pixeli + proporțional)
CROP_STRATEGIES = [
    {"name": "logo_left_tight",   "type": "pixel",        "x": 0, "y": 0, "w": 350, "h": 200},
    {"name": "logo_center_tight", "type": "proportional", "x1": 0.35, "y1": 0.00, "x2": 0.65, "y2": 0.25},
    {"name": "center_login_box",  "type": "proportional", "x1": 0.30, "y1": 0.15, "x2": 0.70, "y2": 0.65},
    {"name": "upper_third_wide",  "type": "proportional", "x1": 0.00, "y1": 0.00, "x2": 1.00, "y2": 0.35},
]

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
    ]},
    {"name": "dropbox", "display": "Dropbox", "category": "tech", "references": [
        {"label": "home",  "url": "https://www.dropbox.com/"},
    ]},
    {"name": "adobe", "display": "Adobe", "category": "tech", "references": [
        {"label": "home",  "url": "https://www.adobe.com/ro/"},
    ]},
    {"name": "zoom", "display": "Zoom", "category": "tech", "references": [
        {"label": "home",  "url": "https://zoom.us/ro-ro/"},
    ]},
    # ── Global — Social ──────────────────────────────────────────────────────
    {"name": "facebook", "display": "Facebook", "category": "social", "references": [
        {"label": "home",  "url": "https://www.facebook.com/"},
    ]},
    {"name": "instagram", "display": "Instagram", "category": "social", "references": [
        {"label": "home",  "url": "https://www.instagram.com/"},
        {"label": "login", "url": "https://www.instagram.com/accounts/login/"},
    ]},
    {"name": "linkedin", "display": "LinkedIn", "category": "social", "references": [
        {"label": "home",  "url": "https://www.linkedin.com/"},
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
    ]},
    {"name": "airbnb", "display": "Airbnb", "category": "travel", "references": [
        {"label": "home",  "url": "https://www.airbnb.com.ro/"},
    ]},
    {"name": "booking", "display": "Booking.com", "category": "travel", "references": [
        {"label": "home",  "url": "https://www.booking.com/"},
    ]},
    # ── Global — Finance ─────────────────────────────────────────────────────
    {"name": "paypal", "display": "PayPal", "category": "finance", "references": [
        {"label": "home",  "url": "https://www.paypal.com/"},
        {"label": "login", "url": "https://www.paypal.com/signin"},
    ]},
    {"name": "revolut", "display": "Revolut", "category": "finance", "references": [
        {"label": "home",  "url": "https://www.revolut.com/ro-RO/"},
    ]},
    {"name": "coinbase", "display": "Coinbase", "category": "finance", "references": [
        {"label": "home",  "url": "https://www.coinbase.com/"},
    ]},
    {"name": "binance", "display": "Binance", "category": "finance", "references": [
        {"label": "home",  "url": "https://www.binance.com/"},
    ]},
    # ── Global — Logistics ───────────────────────────────────────────────────
    {"name": "dhl", "display": "DHL", "category": "logistics", "references": [
        {"label": "home", "url": "https://www.dhl.com/"},
    ]},
    # ── Romanian — Banking ───────────────────────────────────────────────────
    {"name": "bcr", "display": "BCR", "category": "banking-ro", "references": [
        {"label": "home",  "url": "https://www.bcr.ro/"},
        {"label": "login", "url": "https://login.bcr.ro/users/login"},
    ]},
    {"name": "bancatransilvania", "display": "Banca Transilvania", "category": "banking-ro", "references": [
        {"label": "home",    "url": "https://www.bancatransilvania.ro/"},
        {"label": "login",  "url": "https://goapp.bancatransilvania.ro/app/auth/login"},
        {"label": "btpay",   "url": "https://btpay.bancatransilvania.ro/"},
        {"label": "btultra", "url": "https://btultra.btrl.ro/btultraweb/_mcologon"},
    ]},
    {"name": "ing", "display": "ING România", "category": "banking-ro", "references": [
        {"label": "home", "url": "https://www.ing.ro/"},
    ]},
    {"name": "brd", "display": "BRD", "category": "banking-ro", "references": [
        {"label": "home", "url": "https://www.brd.ro/"},
        {"label": "login", "url": "https://www.mybrdnet.ro/brdinternetbank/login.html"},
        {"label": "login2", "url": "https://you.brd.ro/gateway/auth-experience/login#/"},
    ]},
    {"name": "raiffeisen", "display": "Raiffeisen Bank", "category": "banking-ro", "references": [
        {"label": "home", "url": "https://www.raiffeisen.ro/"},
    ]},
    {"name": "cecbank", "display": "CEC Bank", "category": "banking-ro", "references": [
        {"label": "home", "url": "https://www.cec.ro/"},
        {"label": "login", "url": "https://home.ceconline.ro/Identity/Login"},
    ]},
    {"name": "bnr", "display": "BNR", "category": "banking-ro", "references": [
        {"label": "home", "url": "https://www.cec.ro/"},
    ]},
    # ── Romanian — E-commerce ────────────────────────────────────────────────
    {"name": "emag", "display": "eMAG", "category": "ecommerce-ro", "references": [
        {"label": "home",  "url": "https://www.emag.ro/"},
    ]},
    {"name": "olx", "display": "OLX", "category": "ecommerce-ro", "references": [
        {"label": "home",  "url": "https://www.olx.ro/"},
    ]},
    {"name": "altex", "display": "Altex", "category": "ecommerce-ro", "references": [
        {"label": "home", "url": "https://www.altex.ro/"},
    ]},
    {"name": "elefant", "display": "Elefant", "category": "ecommerce-ro", "references": [
        {"label": "home",  "url": "https://www.elefant.ro/"},
    ]},
    {"name": "dedeman", "display": "Dedeman", "category": "ecommerce-ro", "references": [
        {"label": "home",  "url": "https://www.dedeman.ro/"},
    ]},
    {"name": "kaufland", "display": "Kaufland", "category": "retail-ro", "references": [
        {"label": "home", "url": "https://www.kaufland.ro/"},
    ]},
    {"name": "zalando", "display": "Zalando", "category": "ecommerce-ro", "references": [
        {"label": "home", "url": "https://www.zalando.ro/"},
    ]},
    {"name": "fashiondays", "display": "Fashion Days", "category": "ecommerce-ro", "references": [
        {"label": "home",  "url": "https://www.fashiondays.ro/"},
    ]},
    {"name": "aboutyou", "display": "About You", "category": "ecommerce-ro", "references": [
        {"label": "home",  "url": "https://www.aboutyou.ro/"},
    ]},
    # ── Romanian — Logistics ─────────────────────────────────────────────────
    {"name": "postaromana", "display": "Poșta Română", "category": "logistics-ro", "references": [
        {"label": "home",  "url": "https://www.posta-romana.ro/"},
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
    ]},
    # ── Romanian — Gov ───────────────────────────────────────────────────────
    {"name": "anaf", "display": "ANAF", "category": "gov-ro", "references": [
        {"label": "home", "url": "https://www.anaf.ro/"},
    ]},
    {"name": "ghiseulro", "display": "Ghișeul.ro", "category": "gov-ro", "references": [
        {"label": "home",  "url": "https://www.ghiseul.ro/"},
    ]},
    {"name": "cnpp", "display": "CNPP", "category": "gov-ro", "references": [
        {"label": "home",  "url": "https://www.cnpp.ro/"},
        {"label": "login", "url": "https://www.cnpp.ro/autentificare"},
    ]},
    {"name": "roeid", "display": "ROeID", "category": "gov-ro", "references": [        
        {"label": "home", "url": "https://www.roeid.ro/"},    
    ]},
    {"name": "cnas", "display": "CNAS", "category": "gov-ro", "references": [
        {"label": "home",  "url": "https://cnas.ro/"},
        {"label": "login", "url": "https://simsass.cnas.ro/login.php"},
    ]},
    # ── Romanian — Delivery ──────────────────────────────────────────────────
    {"name": "wolt", "display": "Wolt", "category": "delivery-ro", "references": [
        {"label": "home", "url": "https://wolt.com/ro/rou"},
    ]},
    {"name": "glovo", "display": "Glovo", "category": "delivery-ro", "references": [
        {"label": "home",  "url": "https://glovoapp.com/ro/ro/"},     
    ]},
    {"name": "bolt", "display": "Bolt", "category": "delivery-ro", "references": [
        {"label": "home", "url": "https://bolt.eu/ro/"},
    ]},
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

def load_clip_model():
    print("Loading CLIP ViT-B/32...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval()
    print("CLIP model ready\n")
    return model, preprocess


def trim_white_borders(im: Image.Image) -> Optional[Image.Image]:
    """
    Elimină marginile albe. Returnează None dacă imaginea este 100% albă.
    """
    bg = Image.new(im.mode, im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return None


def compute_multi_crop_embeddings(model, preprocess, image_path: str) -> list:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"    [FAIL] image load: {e}")
        return []
 
    w, h = img.size
    embeddings = []
    for strategy in CROP_STRATEGIES:
        
        # Selectăm zona în funcție de tipul strategiei
        if strategy.get("type") == "pixel":
            left = strategy.get("x", 0)
            top = strategy.get("y", 0)
            right = min(w, left + strategy.get("w", w))
            bottom = min(h, top + strategy.get("h", h))
        else:
            left   = max(0, min(w, int(round(strategy["x1"] * w))))
            top    = max(0, min(h, int(round(strategy["y1"] * h))))
            right  = max(0, min(w, int(round(strategy["x2"] * w))))
            bottom = max(0, min(h, int(round(strategy["y2"] * h))))
            
        if right <= left or bottom <= top:
            continue
            
        crop = img.crop((left, top, right, bottom))
        
        # Tăiem spațiul alb (sau ignorăm imaginea dacă e pur albă)
        crop = trim_white_borders(crop)
        
        if crop is None:
            # Sărim peste crop dacă este complet gol! (Asta salvează performanță și previne fals-pozitivele)
            continue
            
        try:
            tensor = preprocess(crop).unsqueeze(0)
            with torch.no_grad():
                emb = model.encode_image(tensor)
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
                {"name": "OptanonConsent", "value": "isGpcEnabled=0&interactionCount=1", "domain": root, "path": "/"},
                {"name": "CookieConsent", "value": "{stamp:'accepted',necessary:true}", "domain": root, "path": "/"},
                {"name": "cookieconsent_status", "value": "dismiss", "domain": root, "path": "/"},
                {"name": "cookie_consent", "value": "accepted", "domain": root, "path": "/"},
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

    for frame in page.frames:
        if frame == page.main_frame:
            continue
        for sel in COOKIE_SELECTORS:
            try:
                locator = frame.locator(sel).first
                if await locator.is_visible(timeout=400):
                    await locator.click(timeout=400)
                    await page.wait_for_timeout(500)
                    return
            except Exception:
                continue

    try:
        await page.evaluate(f"""
            () => {{
                const texts = {COOKIE_TEXTS};
                const els = [
                    ...document.querySelectorAll('div[role="button"]'),
                    ...document.querySelectorAll('button'),
                    ...document.querySelectorAll('a[role="button"]'),
                ];
                for (const el of els) {{
                    if (texts.some(t => el.textContent.trim().includes(t))) {{
                        el.click(); return true;
                    }}
                }}
                return false;
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
                existing = next((r for r in existing_refs if r["label"] == label), None)
                if existing and "embeddings" in existing:
                    print(f"  [{label}] skip — multi-crop exists")
                    continue
                elif existing:
                    shot = existing.get("screenshot")
                    if shot and Path(shot).exists():
                        print(f"  [{label}] upgrading to multi-crop...")
                        embs = compute_multi_crop_embeddings(model, preprocess, shot)
                        if embs:
                            existing["embeddings"] = embs
                            existing.pop("embedding", None)
                            existing.pop("crop_height", None)
                            changed = True
                            print(f"  [{label}] upgraded — {len(embs)} crops (white fields skipped)")
                    continue

            dest = BRANDS_DIR / f"{name}_{label}.png"
            ok = await capture(ref["url"], dest)
            if not ok:
                await asyncio.sleep(2)
                continue

            embs = compute_multi_crop_embeddings(model, preprocess, str(dest))
            if embs:
                new_refs.append({
                    "label": label,
                    "url": ref["url"],
                    "screenshot": str(dest),
                    "embeddings": embs,
                })
                print(f"  [{label}] OK — {len(embs)} crop embeddings (white fields skipped)")
                changed = True
            else:
                print(f"  [{label}] embedding error or completely white")

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
    print(f"\nDone: {processed}/{total} brands, {total_refs} total references")
    print(f"Saved to: {EMBEDDINGS_PATH}")

if __name__ == "__main__":
    asyncio.run(main())