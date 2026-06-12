import uuid
from pathlib import Path
from urllib.parse import urlparse
from playwright.async_api import async_playwright, TimeoutError as PwTimeout
from playwright_stealth import stealth_async

SCREENSHOTS_DIR = Path("/app/screenshots")

# ---------------------------------------------------------------------------
# 1) CSS care ascunde bannerele celor mai răspândite CMP-uri.
#    Injectat înainte de orice script al paginii prin add_init_script,
#    deci e activ în momentul în care bannerul încearcă să se afișeze.
# ---------------------------------------------------------------------------
HIDE_COOKIES_CSS = """
/* OneTrust */
#onetrust-consent-sdk, #onetrust-banner-sdk, .onetrust-pc-dark-filter,
#ot-sdk-btn-floating, .ot-sdk-container,
/* Cookiebot */
#CybotCookiebotDialog, #CybotCookiebotDialogBodyUnderlay,
#CybotCookiebotDialogPoweredByExplanation,
/* TrustArc */
#truste-consent-track, .truste_box_overlay, .truste_overlay,
#truste-consent-content,
/* Quantcast */
.qc-cmp2-container, .qc-cmp-ui-container,
/* Cookie Consent (insites) */
.cc-window, .cc-banner, .cc-overlay,
/* Funding Choices (Google) */
.fc-consent-root, .fc-dialog-overlay, .fc-dialog-container,
/* Didomi */
#didomi-host, #didomi-popup, .didomi-popup-container,
/* Osano */
.osano-cm-window, .osano-cm-dialog,
/* Complianz */
.cmplz-cookiebanner, .cmplz-modal, #cmplz-cookiebanner-container,
/* Iubenda */
.iubenda-cs-container, #iubenda-cs-banner, .iub-cs-container,
/* Termly */
.termly-styled-banner, #termly-consent-banner,
/* Evidon */
.evidon-banner, .evidon-popup-wrapper, #_evidon_banner,
/* Usercentrics */
#usercentrics-root, .uc-banner, [id^="uc-"][class*="banner"],
/* Sourcepoint */
.sp_message_container, .sp_veil, #sp_message_container,
/* Klaro */
.klaro, .cookie-modal,
/* CookieYes */
.cky-consent-container, .cky-overlay,
/* Generic — orice clasă/id care conține "cookie", "consent" sau "gdpr" */
.cookie-notice, .cookie-banner, .cookies-popup, .cookies-modal,
.cookie-consent, .cookie-popup, .cookie-overlay,
[class*="cookie-consent"], [class*="CookieConsent"],
[class*="cookie-banner"], [class*="cookieBanner"],
[class*="cookie-notice"], [class*="cookieNotice"],
[id*="cookie-banner"], [id*="cookieBanner"],
[id*="cookie-consent"], [id*="cookieConsent"],
[id*="cookie-notice"], [id*="cookieNotice"],
[class*="gdpr-banner"], [class*="gdprBanner"], [class*="gdpr-notice"],
[id*="gdpr"],
[aria-label*="cookie" i], [aria-label*="consent" i],
[aria-modal="true"][class*="cookie" i],
/* Variante românești */
[class*="acceptare-cookie"], [class*="cookies-ro"],
[class*="notificare-cookie"]
{
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* Multe site-uri blochează scroll-ul cât bannerul e activ.
   Forțăm body-ul/html-ul la stare normală. */
html, body {
    overflow: auto !important;
    height: auto !important;
}
body.modal-open, body.no-scroll, body.cookie-open,
body.has-cookie-banner, body[style*="overflow: hidden"],
body[style*="overflow:hidden"] {
    overflow: auto !important;
    position: static !important;
}
"""

# ---------------------------------------------------------------------------
# 2) Domenii ale CMP-urilor. Blocăm cererile către ele ca bannerele să nu
#    se mai poată încărca. Pentru screenshot e ideal; dacă ai nevoie ca
#    site-ul să funcționeze complet după, poți comenta secțiunea de routing.
# ---------------------------------------------------------------------------
CMP_BLOCKED_DOMAINS = [
    "cookielaw.org",                       # OneTrust
    "onetrust.com",
    "cookiebot.com",                       # Cookiebot
    "consensu.org",
    "trustarc.com",                        # TrustArc
    "truste.com",
    "quantcast.com",                       # Quantcast
    "quantcount.com",
    "didomi.io",                           # Didomi
    "iubenda.com",                         # Iubenda
    "osano.com",                           # Osano
    "termly.io",                           # Termly
    "evidon.com",                          # Evidon
    "usercentrics.eu",                     # Usercentrics
    "usercentrics.com",
    "sp-prod.net",                         # Sourcepoint
    "fundingchoicesmessages.google.com",   # Google Funding Choices
    "complianz.io",
    "cookieyes.com",
]


async def _block_cmp_requests(route) -> None:
    if any(d in route.request.url for d in CMP_BLOCKED_DOMAINS):
        await route.abort()
    else:
        await route.continue_()


# ---------------------------------------------------------------------------
# Funcția principală
# ---------------------------------------------------------------------------
async def capture_screenshot(url: str) -> str | None:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4()}.png"
    filepath = SCREENSHOTS_DIR / filename

    # Domeniul root + hostul exact (le folosim ambele la cookies)
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        parts = host.split(".")
        root_domain = "." + ".".join(parts[-2:]) if len(parts) >= 2 else host
    except Exception:
        parsed = None
        host = ""
        root_domain = ""

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-blink-features=AutomationControlled",
                 "--disable-features=Translate",
            ],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            ignore_https_errors=True,
            locale="ro-RO",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            extra_http_headers={
                "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "Bypass-Tunnel-Reminder": "true",
            },
        )

        # ------------------------------------------------------------
        # 3) Init script: injectează CSS-ul de ascundere + setează în
        # localStorage/sessionStorage flag-urile pe care multe site-uri le
        # verifică pentru a decide dacă afișează bannerul.
        # ------------------------------------------------------------
        await context.add_init_script(
            f"""
            (() => {{
                const css = {repr(HIDE_COOKIES_CSS)};
                const inject = () => {{
                    if (document.getElementById('__sc_hide_cookies__')) return;
                    const style = document.createElement('style');
                    style.id = '__sc_hide_cookies__';
                    style.textContent = css;
                    (document.head || document.documentElement).appendChild(style);
                }};
                inject();
                // Re-injectăm dacă <head> încă nu există la prima rulare
                try {{
                    new MutationObserver(inject).observe(
                        document.documentElement,
                        {{ childList: true, subtree: false }}
                    );
                }} catch(e) {{}}

                // Flags comune de consimțământ în storage
                const flags = {{
                    'cookieConsent': 'accepted',
                    'cookies_accepted': 'true',
                    'CookieConsent': 'true',
                    'gdpr-consent': 'accepted',
                    'cookie_consent': 'accepted',
                    'cookielaw': 'accepted',
                    'OptanonAlertBoxClosed': new Date().toISOString(),
                }};
                try {{
                    for (const [k, v] of Object.entries(flags)) {{
                        localStorage.setItem(k, v);
                        sessionStorage.setItem(k, v);
                    }}
                }} catch(e) {{}}
            }})();
            """
        )

        # ------------------------------------------------------------
        # 4) Cookies de consimțământ — pe domeniul root și pe hostul exact.
        # ------------------------------------------------------------
        cookie_values = [
            ("OptanonAlertBoxClosed", "true"),
            ("OptanonConsent", "isGpcEnabled=0&interactionCount=1&consentId=accepted"),
            ("CookieConsent", "{stamp:'accepted',necessary:true,preferences:true,statistics:true,marketing:true}"),
            ("CybotCookiebotDialogConsent", "accepted"),
            ("cookieconsent_status", "dismiss"),
            ("cookie_consent", "accepted"),
            ("cookies_accepted", "true"),
            ("cookie-agreed", "2"),
            ("euconsent-v2", "accepted"),
            ("eupubconsent-v2", "accepted"),
            ("didomi_token", "accepted"),
        ]
        domains_to_seed = {d for d in [root_domain, host] if d}
        cookies_to_add = [
            {"name": n, "value": v, "domain": d, "path": "/"}
            for d in domains_to_seed
            for n, v in cookie_values
        ]
        if cookies_to_add:
            try:
                await context.add_cookies(cookies_to_add)
            except Exception:
                pass

        # ------------------------------------------------------------
        # 5) Routing — blocăm domeniile CMP. Comentează blocul dacă vrei
        # ca site-ul să rămână 100% funcțional (în detrimentul unor bannere).
        # ------------------------------------------------------------
        try:
            await context.route("**/*", _block_cmp_requests)
        except Exception:
            pass

        page = await context.new_page()

        # ------------------------------------------------------------
        # 6) Stealth — păstrat ca să nu fim blocați pe site-urile agresive.
        # ------------------------------------------------------------
        try:
            await stealth_async(page)
        except Exception as e:
            print(f"[browser_capture] stealth failed (continuing): {e}")

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=20_000)

            # Lasă rețeaua să se liniștească, dar nu mai mult de 5s
            try:
                await page.wait_for_load_state("networkidle", timeout=5_000)
            except PwTimeout:
                pass

            await page.wait_for_timeout(1500)

            # 7) Fallback prin click — pentru bannerele care au scăpat de CSS
            await _accept_cookies(page)

            await page.wait_for_timeout(1000)
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


# ---------------------------------------------------------------------------
# 8) Fallback prin click — păstrat din versiunea originală, cu timeouturi mai
# generoase și un final pas JS care traversează inclusiv Shadow DOM.
# ---------------------------------------------------------------------------
COOKIE_SELECTORS = [
    # OneTrust
    "#onetrust-accept-btn-handler",
    # Cookiebot
    "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
    "#CybotCookiebotDialogBodyButtonAccept",
    # TrustArc
    "#truste-consent-button",
    # Quantcast
    ".qc-cmp2-summary-buttons button[mode='primary']",
    # Funding Choices
    ".fc-cta-consent",
    # Usercentrics
    "button[data-testid='uc-accept-all-button']",
    # Osano
    "button.osano-cm-accept-all",
    # Complianz
    "button.cmplz-accept",
    # Iubenda
    ".iubenda-cs-accept-btn",
    # Text românesc / english în butoane și div-uri "button"
    "button:has-text('Permite toate modulele cookie')",
    "div[role='button']:has-text('Permite toate modulele cookie')",
    "button:has-text('Acceptă toate')", "button:has-text('Acceptă')",
    "button:has-text('Sunt de acord')", "button:has-text('De acord')",
    "button:has-text('Continuă')", "button:has-text('Am înțeles')",
    "button:has-text('Accept all')", "button:has-text('Accept All')",
    "button:has-text('Accept cookies')", "button:has-text('I agree')",
    "button:has-text('Agree')", "button:has-text('Allow all')",
    "button:has-text('OK')", "button:has-text('Got it')",
    "div[role='button']:has-text('Allow all cookies')",
    "div[role='button']:has-text('Accept all')",
    "#accept-cookies", "#cookie-accept", ".accept-cookies",
    "[aria-label='Accept cookies']", "[data-gdpr-accept='all']",
]

COOKIE_TEXTS = [
    'Permite toate modulele cookie', 'Allow all cookies',
    'Accept all', 'Accept All', 'Acceptă toate', 'Acceptă tot',
    'I agree', 'Agree', 'Allow all', 'Got it', 'OK',
    'De acord', 'Sunt de acord', 'I accept', 'Accept cookies',
    'Continuă', 'Continue', 'I understand', 'Am înțeles',
]


async def _try_selectors_in_context(ctx) -> bool:
    for sel in COOKIE_SELECTORS:
        try:
            locator = ctx.locator(sel).first
            if await locator.is_visible(timeout=1500):
                await locator.click(timeout=1500)
                await ctx.wait_for_timeout(500)
                return True
        except Exception:
            continue
    return False


async def _accept_cookies(page) -> None:
    # Frame principal
    if await _try_selectors_in_context(page):
        return

    # Iframe-uri (Sourcepoint și alte CMP-uri trăiesc adesea în iframe)
    for frame in page.frames:
        if frame == page.main_frame:
            continue
        try:
            if await _try_selectors_in_context(frame):
                return
        except Exception:
            continue

    # Ultim fallback: JS care caută în DOM-ul light + Shadow DOM
    try:
        await page.evaluate(
            """
            (texts) => {
                const tryClick = (root) => {
                    const els = root.querySelectorAll(
                        'button, div[role="button"], a[role="button"],' +
                        '[class*="cookie"] button, [class*="consent"] button'
                    );
                    for (const el of els) {
                        const t = (el.textContent || '').trim();
                        if (texts.some(x => t.includes(x))) {
                            el.click();
                            return true;
                        }
                    }
                    // Mergem și prin shadow roots
                    const all = root.querySelectorAll('*');
                    for (const node of all) {
                        if (node.shadowRoot) {
                            if (tryClick(node.shadowRoot)) return true;
                        }
                    }
                    return false;
                };
                return tryClick(document);
            }
            """,
            COOKIE_TEXTS,
        )
        await page.wait_for_timeout(500)
    except Exception:
        pass