"""
Screenshot each per-persona poster board (docs/popularity_bias/poster_board.html is the interactive
combined page; this renders the standalone single-persona pages from
`poster_board.py --split` and saves a tight PNG per board for the blog).

Drives a headless Google Chrome with a throwaway profile (so it never collides with the
Playwright-MCP browser lock), then crops the uniform #0d1117 margin to the content bbox
so the output matches the existing ww2/fantasy boards (~1124px wide, no border).

    python tools/poster_board.py --split /tmp/boards
    python tools/shoot_boards.py
"""
import os
import subprocess
import sys

from PIL import Image, ImageChops

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BG = (13, 17, 23)  # --bg #0d1117
BOARDS = sys.argv[1:] or [
    ("the-organized-crime-fanatic",       "crime_poster_board.png"),
    ("the-courtroom-drama-devotee",       "courtroom_poster_board.png"),
    ("the-1950s-creature-feature-fan",    "creature_feature_poster_board.png"),
    ("the-argento-fulci-giallo-cultist",  "giallo_poster_board.png"),
    ("the-1930s-screwball-comedy-fan",    "screwball_poster_board.png"),
    ("the-four-quadrant-blockbuster-fan", "blockbuster_poster_board.png"),
    ("the-modern-sci-fi-fan",             "modern_scifi_poster_board.png"),
]


def shoot(slug, out_png):
    raw = f"/tmp/_raw_{slug}.png"
    # Each board gets a fresh profile dir so a lingering Chrome can't lock the next run.
    try:
        subprocess.run([
            CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
            "--no-first-run", "--no-default-browser-check",
            f"--user-data-dir=/tmp/chrome-shot-{slug}",
            "--window-size=1156,1500",
            "--virtual-time-budget=8000",
            f"--screenshot={raw}",
            f"file:///tmp/boards/{slug}.html",
        ], check=False, capture_output=True, timeout=18)
    except subprocess.TimeoutExpired:
        print(f"  ! {slug}: chrome timed out (screenshot may still be written)")

    if not os.path.exists(raw):
        print(f"  !! {slug}: NO screenshot produced")
        return
    im = Image.open(raw).convert("RGB")
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, BG)).getbbox()
    out = im.crop(bbox) if bbox else im
    out.save(f"docs/popularity_bias/figures/{out_png}")
    print(f"  → docs/popularity_bias/figures/{out_png}  ({out.size[0]}x{out.size[1]})")


if __name__ == "__main__":
    for slug, out in BOARDS:
        shoot(slug, out)
