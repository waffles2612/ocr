"""
Image vs OCR Text Query Tester (Groq + Llama Judge)
-----------------------------------------------------
Usage:
    python qwen_tester.py --image invoice.png --query "What is the total?" --csv results.csv
    python qwen_tester.py --image invoice.png --query "Who is the buyer?" --ocr ocr.json

Requirements:
    pip install pytesseract pillow requests
    config.py must have: API_KEY = "gsk_..."  (Groq key)
"""

import argparse
import base64
import csv
import json
import os
import time
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_tess_dir = r"C:\Program Files\Tesseract-OCR"
if _tess_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _tess_dir + os.pathsep + os.environ.get("PATH", "")

try:
    import pytesseract
    from PIL import Image
    import requests
except ImportError as e:
    print(f"[error] Missing dependency: {e}")
    print("Run: pip install pytesseract pillow requests")
    sys.exit(1)

try:
    from config import API_KEY
except ImportError:
    print("[error] config.py not found. Create it with: API_KEY = 'gsk_...'")
    sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────────

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
JUDGE_MODEL  = "llama-3.3-70b-versatile"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Upload image ───────────────────────────────────────────────────────────────

def upload_image(image_path: str) -> str:
    print("  Uploading image to get public URL...")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    resp = requests.post(
        "https://freeimage.host/api/1/upload",
        data={
            "key": "6d207e02198a847aa98d0a2a901485a5",
            "action": "upload",
            "source": b64,
            "format": "json",
        },
        timeout=30,
    )
    data = resp.json()
    if data.get("status_code") == 200:
        url = data["image"]["url"]
        print(f"  Image URL: {url}")
        return url
    raise Exception(f"Upload failed: {data}")


# ── Tesseract OCR with bbox ────────────────────────────────────────────────────

def run_tesseract(image_path: str, lang: str = "eng") -> str:
    img = Image.open(image_path)
    return pytesseract.image_to_string(img, lang=lang).strip()

def run_tesseract_bbox(image_path: str, lang: str = "eng") -> list:
    """Return list of word-level OCR results with bounding boxes."""
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word:
            continue
        words.append({
            "text":  word,
            "left":  data["left"][i],
            "top":   data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i],
            "conf":  data["conf"][i],
            "block_num": data["block_num"][i],
            "line_num":  data["line_num"][i],
        })
    return words

def save_ocr_json(image_path: str, lang: str, out_path: str):
    """Run Tesseract with bbox and save to JSON for visualization."""
    print(f"  Saving OCR bbox data to {out_path}...")
    words = run_tesseract_bbox(image_path, lang=lang)
    img = Image.open(image_path)
    payload = {
        "image": image_path,
        "image_width":  img.width,
        "image_height": img.height,
        "lang": lang,
        "word_count": len(words),
        "words": words,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  {len(words)} words saved. Run: python visualize_ocr.py --ocr {out_path} --image {image_path}")


# ── Call Groq ──────────────────────────────────────────────────────────────────

def call_groq(messages: list, model: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {"model": model, "messages": messages}

    start = time.time()
    try:
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
        elapsed = round((time.time() - start) * 1000)
        data = resp.json()

        if not resp.ok:
            err = data.get("error", {}).get("message", f"HTTP {resp.status_code}")
            return {"text": None, "latency_ms": elapsed, "error": err}

        choices = data.get("choices")
        if not choices:
            return {"text": None, "latency_ms": elapsed,
                    "error": f"No choices. Raw: {json.dumps(data)[:300]}"}

        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            text = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        else:
            text = str(content)

        return {"text": text.strip() or "(empty)", "latency_ms": elapsed, "error": None}

    except requests.exceptions.Timeout:
        elapsed = round((time.time() - start) * 1000)
        return {"text": None, "latency_ms": elapsed, "error": "Timed out (60s)"}
    except Exception as e:
        elapsed = round((time.time() - start) * 1000)
        return {"text": None, "latency_ms": elapsed, "error": str(e)}


# ── Judge ──────────────────────────────────────────────────────────────────────

def judge_answers(query: str, img_answer: str, ocr_answer: str) -> dict:
    prompt = f"""You are a strict answer comparison judge.

Question: {query}

Answer A (from image): {img_answer}
Answer B (from OCR text): {ocr_answer}

Are these two answers semantically equivalent? Consider them equivalent if they convey the same information even if worded differently (e.g. "₹44,000" and "44000.00 INR" are equivalent).

Reply in this exact JSON format with no extra text:
{{"verdict": "MATCH" or "MISMATCH", "reason": "one sentence explanation"}}"""

    result = call_groq([{"role": "user", "content": prompt}], model=JUDGE_MODEL)

    if result["error"]:
        return {"verdict": "ERROR", "reason": result["error"]}

    try:
        text = result["text"].strip().strip("```json").strip("```").strip()
        parsed = json.loads(text)
        return {
            "verdict": parsed.get("verdict", "ERROR"),
            "reason":  parsed.get("reason", "")
        }
    except Exception:
        return {"verdict": "ERROR", "reason": f"Could not parse: {result['text'][:100]}"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def divider(char="─", width=60):
    print(char * width)

def print_result(label: str, result: dict):
    divider()
    print(f"  {label}")
    divider()
    print(f"  Latency : {result['latency_ms']:,} ms")
    if result["error"]:
        print(f"  Error   : {result['error']}")
    else:
        print(f"\n{result['text']}\n")

def save_csv(path: str, row: dict):
    file_exists = Path(path).exists()
    fieldnames = [
        "timestamp", "model", "image", "image_url", "query",
        "ocr_extracted_text",
        "image_answer", "image_latency_ms", "image_error",
        "ocr_answer",   "ocr_latency_ms",   "ocr_error",
        "faster_mode", "delta_ms",
        "verdict", "verdict_reason"
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global VISION_MODEL
    parser = argparse.ArgumentParser(description="Image vs OCR query tester (Groq + Llama Judge)")
    parser.add_argument("--image",  required=True, help="Path to the image file")
    parser.add_argument("--query",  required=True, help="Question to ask")
    parser.add_argument("--lang",   default="eng", help="Tesseract language (default: eng)")
    parser.add_argument("--model",  default=VISION_MODEL, help=f"Vision model (default: {VISION_MODEL})")
    parser.add_argument("--csv",    help="Append result row to CSV file")
    parser.add_argument("--ocr",    help="Save OCR bbox data to this JSON file (for visualization)")
    args = parser.parse_args()
    VISION_MODEL = args.model

    if not Path(args.image).exists():
        print(f"[error] Image not found: {args.image}")
        sys.exit(1)

    print(f"\n  Vision model : {VISION_MODEL}")
    print(f"  Judge model  : {JUDGE_MODEL}")
    print(f"  Image        : {args.image}")
    print(f"  Query        : {args.query}\n")

    # Step 1: Tesseract OCR
    divider("═")
    print("  Step 1 — Running Tesseract OCR...")
    divider("═")
    ocr_text = run_tesseract(args.image, lang=args.lang)
    print(ocr_text[:500] + ("..." if len(ocr_text) > 500 else "") if ocr_text else "  (no text)")

    # Save OCR bbox JSON if requested
    if args.ocr:
        save_ocr_json(args.image, args.lang, args.ocr)

    # Step 2: Upload image
    divider("═")
    print("  Step 2 — Uploading image...")
    divider("═")
    image_url = None
    try:
        image_url = upload_image(args.image)
    except Exception as e:
        print(f"  [warning] Upload failed: {e} — image mode will be skipped.")

    # Step 3: Build messages
    img_messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": args.query}
        ]
    }] if image_url else None

    ocr_messages = [{
        "role": "user",
        "content": (
            f"The following text was extracted via OCR from an image:\n\n"
            f"{ocr_text}\n\n"
            f"Using only this text, answer:\n{args.query}"
        )
    }]

    # Step 4: Query with image
    divider("═")
    print("  Step 3 — Querying with IMAGE...")
    divider("═")
    if img_messages:
        img_result = call_groq(img_messages, VISION_MODEL)
    else:
        img_result = {"text": None, "latency_ms": 0, "error": "Skipped (upload failed)"}
    print_result("IMAGE MODE", img_result)

    # Step 5: Query with OCR text
    divider("═")
    print("  Step 4 — Querying with OCR TEXT...")
    divider("═")
    ocr_result = call_groq(ocr_messages, VISION_MODEL)
    print_result("OCR TEXT MODE", ocr_result)

    # Step 6: Judge
    divider("═")
    print("  Step 5 — Judging answers...")
    divider("═")
    if img_result["error"] or ocr_result["error"]:
        judgment = {"verdict": "SKIP", "reason": "One or both answers errored out"}
    else:
        judgment = judge_answers(args.query, img_result["text"], ocr_result["text"])

    verdict_icon = (
        "✅ MATCH"    if judgment["verdict"] == "MATCH"    else
        "❌ MISMATCH" if judgment["verdict"] == "MISMATCH" else
        "⚠️  SKIP"    if judgment["verdict"] == "SKIP"     else
        "⚠️  ERROR"
    )
    print(f"  Verdict : {verdict_icon}")
    print(f"  Reason  : {judgment['reason']}\n")

    # Step 7: Summary
    divider("═")
    print("  SUMMARY")
    divider("═")
    img_ms = img_result["latency_ms"]
    ocr_ms = ocr_result["latency_ms"]
    delta  = abs(img_ms - ocr_ms)
    faster = "Image" if img_ms < ocr_ms else "OCR text"
    print(f"  Image latency    : {img_ms:,} ms")
    print(f"  OCR text latency : {ocr_ms:,} ms")
    print(f"  Delta            : {delta:,} ms  ({faster} mode was faster)")
    print(f"  Verdict          : {verdict_icon}")
    divider("═")
    print()

    # Step 8: Save CSV
    if args.csv:
        save_csv(args.csv, {
            "timestamp":          time.strftime("%Y-%m-%d %H:%M:%S"),
            "model":              VISION_MODEL,
            "image":              args.image,
            "image_url":          image_url or "",
            "query":              args.query,
            "ocr_extracted_text": ocr_text,
            "image_answer":       img_result.get("text", ""),
            "image_latency_ms":   img_ms,
            "image_error":        img_result.get("error", ""),
            "ocr_answer":         ocr_result.get("text", ""),
            "ocr_latency_ms":     ocr_ms,
            "ocr_error":          ocr_result.get("error", ""),
            "faster_mode":        faster,
            "delta_ms":           delta,
            "verdict":            judgment["verdict"],
            "verdict_reason":     judgment["reason"],
        })
        print(f"  CSV row appended to: {args.csv}\n")


if __name__ == "__main__":
    main()