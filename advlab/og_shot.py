"""Render docs/preview.jpg from the page: one PGD attack plus both decision maps (needs Playwright + Chrome).

    python -m http.server 8000          # from the repository root, in another terminal
    python advlab/og_shot.py
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "preview.jpg"


async def main(url: str) -> None:
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome")
        page = await b.new_page(viewport={"width": 1200, "height": 630}, device_scale_factor=1)
        await page.goto(url + "?lang=en")
        await page.wait_for_function("document.querySelectorAll('#samples [data-sample]').length > 0")
        await page.evaluate("document.getElementById('animate').checked = false")
        await page.click("#go")
        await page.wait_for_function("document.getElementById('stamp').classList.contains('on')")
        await page.click("#map-go")
        await page.wait_for_function("document.getElementById('map-go').textContent.includes('Map')", timeout=180000)
        print("attack:", (await page.inner_text("#metrics")).split("\n")[-1], "· maps:", await page.inner_text("#map-note"))
        # A poster layout: title on the left, the attack and the two maps on the right.
        await page.add_style_tag(content="""
          body{overflow:hidden}.wrap{padding:0}.top,.lede,.warn,.tools,.samples,.controls,.notes,.foot,aside,.lower>section:last-child,.maprow,.keys,.explain,#dview{display:none!important}
          h1{position:fixed;left:48px;top:60px;width:430px;font-size:52px;margin:0}
          .grid{position:fixed;left:520px;top:34px;width:640px;display:block}
          .lower{position:fixed;left:520px;top:330px;width:640px;margin:0;display:block}
          .panel{padding:14px}.maps{grid-template-columns:1fr 1fr}.map{aspect-ratio:2/1}
        """)
        await page.evaluate("""document.body.insertAdjacentHTML('beforeend','<div style="position:fixed;left:48px;bottom:48px;font:500 18px JetBrains Mono,monospace;color:#8b93a1"><b style=color:#eef1f5>niansia.com</b> · adversarial lab</div>')""")
        await page.wait_for_timeout(400)
        png = OUT.with_suffix(".png")
        await page.screenshot(path=str(png))
        await b.close()
    Image.open(png).convert("RGB").save(OUT, quality=86, optimize=True, progressive=True)
    png.unlink()
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/web/")
    asyncio.run(main(ap.parse_args().url))
