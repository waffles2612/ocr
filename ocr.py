import os
import sys

_tess_dir = r"C:\Program Files\Tesseract-OCR"
if _tess_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _tess_dir + os.pathsep + os.environ.get("PATH", "")

try:
    import pytesseract
    from PIL import Image
except ImportError as e:
    raise ImportError(f"Missing dependency: {e}. Run: pip install pytesseract pillow") from e

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
