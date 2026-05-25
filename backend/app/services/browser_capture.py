import uuid
from pathlib import Path
from typing import Optional

SCREENSHOTS_DIR = Path("/app/screenshots")


async def capture_screenshot(url: str, timeout_ms: int = 15_000) -> Optional[str]:
    from playwright.async_api import async_playwright, TimeoutError as PwTimeout

    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4()}.png"
    filepath = SCREENSHOTS_DIR / filename

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-blink-features=AutomationControlled",
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
        page = await context.new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            await _try_accept_cookies(page)
            await page.wait_for_timeout(1500)
            await page.screenshot(path=str(filepath), full_page=False)
        except PwTimeout:
            print(f"[browser_capture] timeout for: {url}")
            _cleanup(filepath)
            return None
        except Exception as exc:
            print(f"[browser_capture] error for {url!r}: {exc}")
            _cleanup(filepath)
            return None
        finally:
            await browser.close()

    return filename


async def _try_accept_cookies(page) -> None:
    selectors = [
        "button:has-text('Accept all')",
        "button:has-text('Accept All')",
        "button:has-text('Accept all cookies')",
        "button:has-text('Accept cookies')",
        "button:has-text('I Accept')",
        "button:has-text('I agree')",
        "button:has-text('Agree')",
        "button:has-text('OK')",
        "button:has-text('Got it')",
        "button:has-text('Allow all')",
        "button:has-text('Acceptă toate')",
        "button:has-text('Accept toate')",
        "button:has-text('Acceptă')",
        "button:has-text('Sunt de acord')",
        "button:has-text('De acord')",
    ]
    for selector in selectors:
        try:
            await page.locator(selector).first.click(timeout=1000)
            await page.wait_for_timeout(500)
            return
        except Exception:
            continue


def _cleanup(filepath: Path) -> None:
    try:
        if filepath.exists():
            filepath.unlink()
    except Exception:
        pass