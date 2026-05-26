import uuid
from pathlib import Path
from urllib.parse import urlparse
from playwright.async_api import async_playwright, TimeoutError as PwTimeout

SCREENSHOTS_DIR = Path("/app/screenshots")


async def capture_screenshot(url: str) -> str | None:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4()}.png"
    filepath = SCREENSHOTS_DIR / filename

    # Extract domain for cookie injection
    try:
        parsed = urlparse(url)
        domain = parsed.hostname or ""
        # Use root domain (e.g. facebook.com not www.facebook.com)
        parts = domain.split(".")
        root_domain = "." + ".".join(parts[-2:]) if len(parts) >= 2 else domain
    except Exception:
        root_domain = ""

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )

        # Inject consent cookies before navigation — prevents most banners from appearing
        if root_domain:
            try:
                await context.add_cookies([
                    # OneTrust (300k+ sites)
                    {"name": "OptanonAlertBoxClosed", "value": "true",
                     "domain": root_domain, "path": "/"},
                    {"name": "OptanonConsent",
                     "value": "isGpcEnabled=0&isIABGlobal=false&consentId=1&interactionCount=1",
                     "domain": root_domain, "path": "/"},
                    # Cookiebot / Usercentrics
                    {"name": "CookieConsent",
                     "value": "{stamp:'accepted',necessary:true,preferences:true,statistics:true,marketing:true}",
                     "domain": root_domain, "path": "/"},
                    # Generic implementations
                    {"name": "cookieconsent_status", "value": "dismiss",
                     "domain": root_domain, "path": "/"},
                    {"name": "cookie_consent", "value": "accepted",
                     "domain": root_domain, "path": "/"},
                    {"name": "gdpr_consent", "value": "true",
                     "domain": root_domain, "path": "/"},
                    {"name": "cookies_accepted", "value": "true",
                     "domain": root_domain, "path": "/"},
                    {"name": "cookie-agreed", "value": "2",
                     "domain": root_domain, "path": "/"},
                    {"name": "euconsent-v2", "value": "accepted",
                     "domain": root_domain, "path": "/"},
                ])
            except Exception:
                pass

        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
            await page.wait_for_timeout(2000)
            # Fallback: try clicking remaining banners not handled by cookies
            await _accept_cookies(page)
            await page.wait_for_timeout(1500)
            await page.screenshot(path=str(filepath), full_page=False)
            return filename
        except PwTimeout:
            print(f"[browser_capture] timeout for '{url}'")
            return None
        except Exception as e:
            print(f"[browser_capture] error for '{url}': {e}")
            return None
        finally:
            await browser.close()


COOKIE_SELECTORS = [
    # OneTrust
    "#onetrust-accept-btn-handler",
    ".onetrust-accept-btn-handler",
    # Cookiebot
    "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
    "#CybotCookiebotDialogBodyButtonAccept",
    # TrustArc
    "#truste-consent-button",
    ".truste-button-2",
    # Quantcast
    ".qc-cmp2-summary-buttons button:first-child",
    # div[role=button] — Facebook, Meta, Google
    "div[role='button']:has-text('Permite toate modulele cookie')",
    "div[role='button']:has-text('Allow all cookies')",
    "div[role='button']:has-text('Accept all')",
    "div[role='button']:has-text('Acceptă toate')",
    "div[role='button']:has-text('I agree')",
    # Standard buttons EN
    "button:has-text('Accept all')",
    "button:has-text('Accept All')",
    "button:has-text('Accept cookies')",
    "button:has-text('I accept')",
    "button:has-text('I agree')",
    "button:has-text('Agree')",
    "button:has-text('Allow all')",
    "button:has-text('Allow All')",
    "button:has-text('OK')",
    "button:has-text('Got it')",
    "button:has-text('Continue')",
    # Standard buttons RO
    "button:has-text('Acceptă toate')",
    "button:has-text('Acceptă')",
    "button:has-text('Sunt de acord')",
    "button:has-text('De acord')",
    "button:has-text('Permite tot')",
    "button:has-text('Continuă')",
    "button:has-text('Am înțeles')",
    # ID / class generice
    "#accept-cookies",
    "#cookie-accept",
    "#cookieAccept",
    "#btn-cookie-allow",
    ".accept-cookies",
    ".cookie-accept",
    ".cc-btn.cc-allow",
    # Atribute
    "[aria-label='Accept cookies']",
    "[aria-label='Accept all cookies']",
    "[data-testid='cookie-accept']",
    "[data-gdpr-accept='all']",
]

COOKIE_TEXTS = [
    'Permite toate modulele cookie', 'Allow all cookies',
    'Accept all', 'Accept All', 'Acceptă toate',
    'I agree', 'Agree', 'Allow all', 'Got it', 'OK',
    'De acord', 'Sunt de acord', 'I accept', 'Accept cookies',
    'Continuă', 'Continue', 'I understand', 'Am înțeles',
]


async def _try_selectors_in_context(ctx) -> bool:
    for sel in COOKIE_SELECTORS:
        try:
            locator = ctx.locator(sel).first
            if await locator.is_visible(timeout=600):
                await locator.click(timeout=600)
                await ctx.wait_for_timeout(500)
                return True
        except Exception:
            continue
    return False


async def _accept_cookies(page) -> None:
    # 1. CSS selectors in main page
    found = await _try_selectors_in_context(page)
    if found:
        return

    # 2. CSS selectors in iframes
    for frame in page.frames:
        if frame == page.main_frame:
            continue
        try:
            found = await _try_selectors_in_context(frame)
            if found:
                return
        except Exception:
            continue

    # 3. JavaScript fallback — catches dynamic banners (Facebook, Meta etc.)
    try:
        await page.evaluate(f"""
            () => {{
                const texts = {COOKIE_TEXTS};
                const els = [
                    ...document.querySelectorAll('div[role="button"]'),
                    ...document.querySelectorAll('button'),
                    ...document.querySelectorAll('a[role="button"]'),
                    ...document.querySelectorAll('[class*="cookie"] button'),
                    ...document.querySelectorAll('[class*="consent"] button'),
                ];
                for (const el of els) {{
                    const text = el.textContent.trim();
                    if (texts.some(t => text.includes(t))) {{
                        el.click();
                        return true;
                    }}
                }}
                return false;
            }}
        """)
        await page.wait_for_timeout(500)
    except Exception:
        pass