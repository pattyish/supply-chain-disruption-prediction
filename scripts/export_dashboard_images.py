from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps
from playwright.sync_api import sync_playwright


TAB_NAMES = [
    "Prediction & What-if",
    "Prescriptive Optimization",
    "Anomaly + Playbooks",
    "Provenance + Feedback",
]


def slugify(name: str) -> str:
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "&", "+"}:
            out.append("_")
    return "".join(out).strip("_")


def capture_tab_screenshots(
    url: str,
    out_dir: Path,
    width: int,
    height: int,
) -> list[Path]:
    screenshots: list[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height})

        response = page.goto(url, wait_until="domcontentloaded", timeout=60000)
        if response is not None and response.status >= 400:
            browser.close()
            raise RuntimeError(f"Dashboard URL returned HTTP {response.status}: {url}")

        # Streamlit DOM can vary across versions; wait for any likely app container.
        page.wait_for_selector("body", state="attached", timeout=60000)
        page.wait_for_function("() => document.readyState === 'interactive' || document.readyState === 'complete'", timeout=60000)

        # Some Streamlit versions temporarily hide body while app bootstraps.
        page.evaluate(
            """
            () => {
                if (document.body) {
                    document.body.style.visibility = 'visible';
                    document.body.style.opacity = '1';
                }
            }
            """
        )

        try:
            page.wait_for_function(
                """
                () => {
                    const selectors = [
                        '[data-testid="stAppViewContainer"]',
                        '[data-testid="stApp"]',
                        '[data-testid="stTabs"]',
                        'section.main'
                    ];
                    return selectors.some((s) => !!document.querySelector(s));
                }
                """,
                timeout=45000,
            )
        except Exception as exc:
            browser.close()
            raise RuntimeError(
                "Dashboard UI container not detected. Ensure Streamlit is running and fully loaded at the provided URL."
            ) from exc

        page.wait_for_timeout(3000)

        def click_tab_by_name(name: str) -> bool:
            candidates = [
                page.get_by_role("tab", name=name),
                page.locator("[role='tab']").filter(has_text=name).first,
                page.locator("button").filter(has_text=name).first,
                page.get_by_text(name, exact=False).first,
            ]

            for locator in candidates:
                try:
                    if locator.count() > 0:
                        locator.first.scroll_into_view_if_needed(timeout=3000)
                        locator.first.click(timeout=5000)
                        return True
                except Exception:
                    continue
            return False

        def click_tab_by_index(tab_index_zero_based: int) -> bool:
            # Streamlit tabs usually expose role=tab; this fallback avoids brittle label matching.
            tab_list = page.locator("[role='tab']")
            try:
                count = tab_list.count()
            except Exception:
                return False

            if count <= tab_index_zero_based:
                return False

            tab = tab_list.nth(tab_index_zero_based)
            try:
                tab.scroll_into_view_if_needed(timeout=3000)
                tab.click(timeout=5000)
                return True
            except Exception:
                try:
                    # Last resort for edge UI overlays.
                    tab.click(timeout=5000, force=True)
                    return True
                except Exception:
                    return False

        for idx, tab_name in enumerate(TAB_NAMES, start=1):
            clicked = click_tab_by_name(tab_name)
            if not clicked:
                clicked = click_tab_by_index(idx - 1)
            if not clicked and idx != 1:
                browser.close()
                raise RuntimeError(
                    f"Could not find/click tab '{tab_name}'. Ensure the dashboard is running and tab names match."
                )

            page.wait_for_timeout(1200)

            img_path = out_dir / f"tab_{idx:02d}_{slugify(tab_name)}.png"
            page.screenshot(path=str(img_path), full_page=True)
            screenshots.append(img_path)

        browser.close()

    return screenshots


def merge_2x2(images: list[Path], output_path: Path) -> None:
    if len(images) != 4:
        raise ValueError("Expected exactly 4 images to compose a 2x2 output.")

    opened = [Image.open(p).convert("RGB") for p in images]

    max_w = max(img.width for img in opened)
    max_h = max(img.height for img in opened)

    padded: list[Image.Image] = []
    for img in opened:
        if img.width != max_w or img.height != max_h:
            pad_left = (max_w - img.width) // 2
            pad_top = (max_h - img.height) // 2
            pad_right = max_w - img.width - pad_left
            pad_bottom = max_h - img.height - pad_top
            img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill="white")
        padded.append(img)

    canvas = Image.new("RGB", (max_w * 2, max_h * 2), "white")
    positions = [(0, 0), (max_w, 0), (0, max_h), (max_w, max_h)]

    for img, pos in zip(padded, positions):
        canvas.paste(img, pos)

    canvas.save(output_path)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Streamlit dashboard tab screenshots and merge into one 2x2 image."
    )
    parser.add_argument("--url", default="http://localhost:8501", help="Dashboard URL")
    parser.add_argument("--out-dir", default="reports/figures", help="Directory for exported images")
    parser.add_argument("--width", type=int, default=1440, help="Browser viewport width")
    parser.add_argument("--height", type=int, default=900, help="Browser viewport height")
    parser.add_argument(
        "--merged-name",
        default="dashboard-full-2x2.png",
        help="Filename for merged output image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    screenshots = capture_tab_screenshots(
        url=args.url,
        out_dir=out_dir,
        width=args.width,
        height=args.height,
    )

    merged_path = out_dir / args.merged_name
    merge_2x2(screenshots, merged_path)

    print("Saved tab screenshots:")
    for p in screenshots:
        print(f"- {p}")
    print(f"Saved merged image: {merged_path}")


if __name__ == "__main__":
    main()
