"""Screenshots and a GIF of the HTML reports, for the README.

    uv run --with playwright --with pillow python scripts/capture_demo.py runs/v1/<id>/report.html runs/v2/<id>/report.html

Needs a Chromium for Playwright (`uv run --with playwright playwright install chromium`).
"""

import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

OUT = Path("docs/img")
SIZE = {"width": 1180, "height": 760}


def main(v1: str, v2: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=SIZE, device_scale_factor=1)

        def shot(name: str, hold: int = 1) -> None:
            path = OUT / f"{name}.png"
            page.screenshot(path=path)
            frames.extend([Image.open(path).convert("RGB")] * hold)

        page.goto(Path(v1).resolve().as_uri())
        shot("v1-report", hold=3)
        page.goto(Path(v2).resolve().as_uri())
        shot("v2-report", hold=3)
        for sid in ("emergency-transfer", "office-hours"):
            row = page.locator("tr.main", has_text=sid).first
            row.locator("xpath=following-sibling::tr[1]//summary").click()
            row.scroll_into_view_if_needed()
            page.evaluate("window.scrollBy(0, -60)")
            shot(f"v2-{sid}", hold=4)
        browser.close()
    frames[0].save(OUT / "demo.gif", save_all=True, append_images=frames[1:], duration=900, loop=0, optimize=True)
    print(f"wrote {len(list(OUT.glob('*.png')))} screenshots and {OUT / 'demo.gif'}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
