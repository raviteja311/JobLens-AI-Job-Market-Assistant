"""Record the README demo GIF from the running Streamlit UI.

    docker compose up -d db api ui
    python scripts/demo_gif.py            # writes docs/img/demo.gif

Drives the real UI in a headless browser and stitches screenshots into a GIF,
so the demo shows what the deployed app actually renders rather than a
mockup. Uses the system Edge through Playwright to avoid a browser download;
pass --channel chromium if you have Playwright's own browsers installed.

Only the Search and Trends tabs are recorded. Chat and resume matching need
an LLM backend and take about 50 seconds an answer on a local model, which
is a fact for the README, not a frame for a GIF.
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img" / "demo.gif"

QUERIES = [
    "remote machine learning engineer",
    "rust backend for a fintech",
    "pgvector",
]


def shot(page, frames, hold: int = 1, width: int = 1100) -> None:
    """Append one screenshot, repeated `hold` times so it stays on screen."""
    png = page.screenshot(full_page=False)
    image = Image.open(io.BytesIO(png)).convert("RGB")
    if image.width != width:
        ratio = width / image.width
        image = image.resize((width, int(image.height * ratio)), Image.LANCZOS)
    for _ in range(hold):
        frames.append(image)


def record(url: str, channel: str | None) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with sync_playwright() as p:
        launch = {"headless": True}
        if channel:
            launch["channel"] = channel
        browser = p.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": 1280, "height": 860})
        page.goto(url, wait_until="networkidle")
        page.wait_for_selector("text=Search", timeout=60_000)
        time.sleep(2)
        shot(page, frames, hold=3)

        box = page.get_by_label("Search", exact=True)
        for query in QUERIES:
            box.fill("")
            for ch in query:
                box.type(ch, delay=35)
            box.press("Enter")
            page.wait_for_timeout(2500)
            shot(page, frames, hold=5)

        page.get_by_role("tab", name="Trends").click()
        page.wait_for_timeout(3000)
        shot(page, frames, hold=6)
        page.mouse.wheel(0, 600)
        page.wait_for_timeout(1200)
        shot(page, frames, hold=5)

        browser.close()
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--channel", default="msedge")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    frames = record(args.url, args.channel or None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=600,
        loop=0,
        optimize=True,
    )
    size_kb = args.out.stat().st_size // 1024
    print(f"wrote {args.out} ({len(frames)} frames, {size_kb} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
