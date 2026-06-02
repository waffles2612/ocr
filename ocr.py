"""
OCR Module — runs Tesseract on an image and saves results to JSON.

Usage:
    python ocr.py --image invoice.png --out ocr.json
    python ocr.py --image invoice.png --out ocr.json --lang eng
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_tess_dir = r"C:\Program Files\Tesseract-OCR"
if _tess_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _tess_dir + os.pathsep + os.environ.get("PATH", "")

try:
    import pytesseract
    from PIL import Image
except ImportError as e:
    print(f"[error] Missing dependency: {e}")
    print("Run: pip install pytesseract pillow")
    sys.exit(1)

if sys.platform == "win32":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def run_ocr(image_path: str, lang: str = "eng") -> dict:
    img = Image.open(image_path)

    plain_text = pytesseract.image_to_string(img, lang=lang).strip()

    raw = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(raw["text"])):
        word = raw["text"][i].strip()
        if not word:
            continue
        words.append({
            "text":      word,
            "left":      raw["left"][i],
            "top":       raw["top"][i],
            "width":     raw["width"][i],
            "height":    raw["height"][i],
            "conf":      raw["conf"][i],
            "block_num": raw["block_num"][i],
            "line_num":  raw["line_num"][i],
        })

    return {
        "image":        image_path,
        "image_width":  img.width,
        "image_height": img.height,
        "lang":         lang,
        "plain_text":   plain_text,
        "word_count":   len(words),
        "words":        words,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Tesseract OCR and save results to JSON")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--out",   required=True, help="Path to save OCR JSON output")
    parser.add_argument("--lang",  default="eng", help="Tesseract language code (default: eng)")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"[error] Image not found: {args.image}")
        sys.exit(1)

    print(f"\n  Image : {args.image}")
    print(f"  Lang  : {args.lang}")
    print(f"  Out   : {args.out}\n")

    result = run_ocr(args.image, lang=args.lang)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  Extracted {result['word_count']} words.")
    print(f"  Plain text preview:\n")
    preview = result["plain_text"][:400]
    print(preview + ("..." if len(result["plain_text"]) > 400 else ""))
    print(f"\n  Saved to: {args.out}\n")


if __name__ == "__main__":
    main()
