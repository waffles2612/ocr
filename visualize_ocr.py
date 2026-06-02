"""
OCR Bounding Box Visualizer
-----------------------------
Called automatically by qwen_tester.py, or run standalone:
    python visualize_ocr.py --ocr ocr.json --image invoice.png --out annotated.png --min-conf 60
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("[error] pip install pillow")
    sys.exit(1)


def draw_boxes(image_path: str, words: list, min_conf: int = 0) -> tuple:
    """Draw bounding boxes on the image. Returns (PIL.Image, drawn_count)."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    drawn = 0
    for w in words:
        conf = int(w.get("conf", -1))
        if conf < min_conf:
            continue

        x, y, width, height = w["left"], w["top"], w["width"], w["height"]

        if conf >= 80:
            color = (34, 197, 94)
        elif conf >= 50:
            color = (251, 146, 60)
        else:
            color = (239, 68, 68)

        draw.rectangle([x, y, x + width, y + height], outline=color, width=2)
        draw.text((x, max(0, y - 13)), w["text"], fill=color, font=font)
        drawn += 1

    return img, drawn


def visualize(ocr_json: str, image_path: str, out_path: str = "annotated.png", min_conf: int = 0):
    with open(ocr_json, encoding="utf-8") as f:
        data = json.load(f)

    words = data.get("words", [])
    img, drawn = draw_boxes(image_path, words, min_conf=min_conf)
    img.save(out_path)

    print(f"  Annotated image saved: {out_path}")
    print(f"  Words drawn : {drawn} / {len(words)}")
    print(f"  Color guide : green=high conf (≥80), orange=medium (50-79), red=low (<50)")


def main():
    parser = argparse.ArgumentParser(description="Visualize OCR bounding boxes")
    parser.add_argument("--ocr",      required=True, help="Path to OCR JSON file")
    parser.add_argument("--image",    required=True, help="Path to original image")
    parser.add_argument("--out",      default="annotated.png", help="Output image (default: annotated.png)")
    parser.add_argument("--min-conf", type=int, default=0, help="Minimum confidence to show (default: 0)")
    args = parser.parse_args()

    if not Path(args.ocr).exists():
        print(f"[error] OCR file not found: {args.ocr}")
        sys.exit(1)
    if not Path(args.image).exists():
        print(f"[error] Image not found: {args.image}")
        sys.exit(1)

    with open(args.ocr, encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n  Image      : {args.image}")
    print(f"  OCR file   : {args.ocr}")
    print(f"  Total words: {len(data.get('words', []))}\n")

    with open(args.ocr, encoding="utf-8") as f:
        data = json.load(f)
    img, drawn = draw_boxes(args.image, data.get("words", []), min_conf=args.min_conf)
    img.save(args.out)
    print(f"  Drew {drawn} / {len(data.get('words', []))} words.")
    print(f"  Color guide : green=high conf (≥80), orange=medium (50-79), red=low (<50)")
    print(f"\n  Done! Open {args.out}\n")


if __name__ == "__main__":
    main()
