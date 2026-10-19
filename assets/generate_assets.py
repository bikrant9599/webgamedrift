"""
Generate sleek racing app icons for Burnout Camera Drive.
"""

import os
from PIL import Image, ImageDraw

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
PNG_PATH = os.path.join(ASSETS_DIR, "app_icon.png")
ICO_PATH = os.path.join(ASSETS_DIR, "app_icon.ico")


def create_racing_icon():
    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Dark rounded background shield
    pad = 12
    draw.rounded_rectangle([pad, pad, size - pad, size - pad], radius=48, fill=(18, 20, 24, 255), outline=(0, 210, 255, 255), width=6)

    # Inner circular steering rim
    cx, cy = size // 2, size // 2
    r_outer = 85
    r_inner = 65

    # Steering wheel outer rim
    draw.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], outline=(0, 210, 255, 255), width=16)

    # Center hub
    r_hub = 26
    draw.ellipse([cx - r_hub, cy - r_hub, cx + r_hub, cy + r_hub], fill=(0, 240, 255, 255), outline=(255, 255, 255, 255), width=4)

    # 3 Steering Spokes
    # Left spoke
    draw.line([cx - r_hub, cy, cx - r_outer + 8, cy], fill=(0, 210, 255, 255), width=12)
    # Right spoke
    draw.line([cx + r_hub, cy, cx + r_outer - 8, cy], fill=(0, 210, 255, 255), width=12)
    # Bottom vertical spoke
    draw.line([cx, cy + r_hub, cx, cy + r_outer - 8], fill=(0, 210, 255, 255), width=12)

    # Camera lens indicator at top center
    r_cam = 10
    draw.ellipse([cx - r_cam, cy - r_outer - 18, cx + r_cam, cy - r_outer + 2], fill=(255, 50, 100, 255), outline=(255, 255, 255, 255), width=3)

    img.save(PNG_PATH, "PNG")
    img.save(ICO_PATH, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"Generated icons at {PNG_PATH} and {ICO_PATH}")


if __name__ == "__main__":
    create_racing_icon()
