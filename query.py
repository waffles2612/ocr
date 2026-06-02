"""
Query Module — reads OCR JSON, sends to LLM (image + OCR text modes), saves CSV.

Usage:
    python query.py --ocr ocr.json --query "What is the total?" --csv results.csv
    python query.py --ocr ocr.json --query "Who is the buyer?"  --model meta-llama/llama-4-scout-17b-16e-instruct
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

try:
    import requests
except ImportError:
    print("[error] Missing dependency: requests")
    print("Run: pip install requests")
    sys.exit(1)

try:
    from config import API_KEY
except ImportError:
    API_KEY = None  # will be injected at runtime by app.py

# ── Config ─────────────────────────────────────────────────────────────────────

VISION_MODEL       = "meta-llama/llama-4-scout-17b-16e-instruct"
JUDGE_MODEL        = "llama-3.3-70b-versatile"
GROQ_URL           = "https://api.groq.com/openai/v1/chat/completions"
JUDGE_GEMINI_MODEL = "gemini-2.5-flash"

# ── Helpers ────────────────────────────────────────────────────────────────────

def is_gemini(model: str) -> bool:
    return model.startswith("gemini")


def call_gemini(content, model: str, api_key: str) -> dict:
    try:
        import google.generativeai as genai
    except ImportError:
        return {"text": None, "latency_ms": 0, "error": "google-generativeai not installed"}
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model)
    start = time.time()
    try:
        response = client.generate_content(content)
        elapsed = round((time.time() - start) * 1000)
        return {"text": response.text.strip(), "latency_ms": elapsed, "error": None}
    except Exception as e:
        elapsed = round((time.time() - start) * 1000)
        return {"text": None, "latency_ms": elapsed, "error": str(e)}


def upload_image(image_path: str) -> str:
    print("  Uploading image to get public URL...")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    resp = requests.post(
        "https://freeimage.host/api/1/upload",
        data={"key": "6d207e02198a847aa98d0a2a901485a5", "action": "upload",
              "source": b64, "format": "json"},
        timeout=30,
    )
    data = resp.json()
    if data.get("status_code") == 200:
        url = data["image"]["url"]
        print(f"  Image URL: {url}")
        return url
    raise Exception(f"Upload failed: {data}")


def call_groq(messages: list, model: str) -> dict:
    import query as _self
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_self.API_KEY}",
    }
    start = time.time()
    try:
        resp = requests.post(GROQ_URL, headers=headers,
                             json={"model": model, "messages": messages}, timeout=60)
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
        text = (
            "".join(c.get("text", "") for c in content if isinstance(c, dict))
            if isinstance(content, list) else str(content)
        )
        return {"text": text.strip() or "(empty)", "latency_ms": elapsed, "error": None}

    except requests.exceptions.Timeout:
        elapsed = round((time.time() - start) * 1000)
        return {"text": None, "latency_ms": elapsed, "error": "Timed out (60s)"}
    except Exception as e:
        elapsed = round((time.time() - start) * 1000)
        return {"text": None, "latency_ms": elapsed, "error": str(e)}


def judge_answers(query: str, img_answer: str, ocr_answer: str, gemini_key: str = None) -> dict:
    prompt = f"""You are a strict answer comparison judge.

Question: {query}

Answer A (from image): {img_answer}
Answer B (from OCR text): {ocr_answer}

Are these two answers semantically equivalent? Consider them equivalent if they convey the same information even if worded differently (e.g. "₹44,000" and "44000.00 INR" are equivalent).

Reply in this exact JSON format with no extra text:
{{"verdict": "MATCH" or "MISMATCH", "reason": "one sentence explanation"}}"""

    if gemini_key:
        result = call_gemini(prompt, JUDGE_GEMINI_MODEL, gemini_key)
    else:
        result = call_groq([{"role": "user", "content": prompt}], model=JUDGE_MODEL)
    if result["error"]:
        return {"verdict": "ERROR", "reason": result["error"]}
    try:
        text = result["text"].strip().strip("```json").strip("```").strip()
        parsed = json.loads(text)
        return {"verdict": parsed.get("verdict", "ERROR"), "reason": parsed.get("reason", "")}
    except Exception:
        return {"verdict": "ERROR", "reason": f"Could not parse: {result['text'][:100]}"}


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
        "verdict", "verdict_reason",
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global VISION_MODEL
    parser = argparse.ArgumentParser(description="Query LLM with image and OCR text modes")
    parser.add_argument("--ocr",   required=True, help="Path to OCR JSON file (from ocr.py)")
    parser.add_argument("--query", required=True, help="Question to ask about the image")
    parser.add_argument("--model", default=VISION_MODEL, help=f"Vision model (default: {VISION_MODEL})")
    parser.add_argument("--csv",   help="Append result row to CSV file")
    args = parser.parse_args()
    VISION_MODEL = args.model

    if not Path(args.ocr).exists():
        print(f"[error] OCR file not found: {args.ocr}")
        print("Run: python ocr.py --image <image> --out <ocr.json>")
        sys.exit(1)

    with open(args.ocr, encoding="utf-8") as f:
        ocr_data = json.load(f)

    image_path = ocr_data["image"]
    ocr_text   = ocr_data["plain_text"]

    print(f"\n  Vision model : {VISION_MODEL}")
    print(f"  Judge model  : {JUDGE_MODEL}")
    print(f"  Image        : {image_path}")
    print(f"  OCR file     : {args.ocr}")
    print(f"  Query        : {args.query}\n")

    # Step 1: Upload image
    divider("═")
    print("  Step 1 — Uploading image...")
    divider("═")
    image_url = None
    try:
        image_url = upload_image(image_path)
    except Exception as e:
        print(f"  [warning] Upload failed: {e} — image mode will be skipped.")

    # Step 2: Query with image
    divider("═")
    print("  Step 2 — Querying with IMAGE...")
    divider("═")
    if image_url:
        img_messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": args.query},
        ]}]
        img_result = call_groq(img_messages, VISION_MODEL)
    else:
        img_result = {"text": None, "latency_ms": 0, "error": "Skipped (upload failed)"}
    print_result("IMAGE MODE", img_result)

    # Step 3: Query with OCR text
    divider("═")
    print("  Step 3 — Querying with OCR TEXT...")
    divider("═")
    ocr_messages = [{"role": "user", "content": (
        f"The following text was extracted via OCR from an image:\n\n"
        f"{ocr_text}\n\n"
        f"Using only this text, answer:\n{args.query}"
    )}]
    ocr_result = call_groq(ocr_messages, VISION_MODEL)
    print_result("OCR TEXT MODE", ocr_result)

    # Step 4: Judge
    divider("═")
    print("  Step 4 — Judging answers...")
    divider("═")
    if img_result["error"] or ocr_result["error"]:
        judgment = {"verdict": "SKIP", "reason": "One or both answers errored out"}
    else:
        judgment = judge_answers(args.query, img_result["text"], ocr_result["text"])

    verdict_icon = (
        "MATCH"    if judgment["verdict"] == "MATCH"    else
        "MISMATCH" if judgment["verdict"] == "MISMATCH" else
        "SKIP"     if judgment["verdict"] == "SKIP"     else
        "ERROR"
    )
    print(f"  Verdict : {verdict_icon}")
    print(f"  Reason  : {judgment['reason']}\n")

    # Step 5: Summary
    img_ms = img_result["latency_ms"]
    ocr_ms = ocr_result["latency_ms"]
    delta  = abs(img_ms - ocr_ms)
    faster = "Image" if img_ms < ocr_ms else "OCR text"

    divider("═")
    print("  SUMMARY")
    divider("═")
    print(f"  Image latency    : {img_ms:,} ms")
    print(f"  OCR text latency : {ocr_ms:,} ms")
    print(f"  Delta            : {delta:,} ms  ({faster} mode was faster)")
    print(f"  Verdict          : {verdict_icon}")
    divider("═")
    print()

    # Step 6: Save CSV
    if args.csv:
        save_csv(args.csv, {
            "timestamp":          time.strftime("%Y-%m-%d %H:%M:%S"),
            "model":              VISION_MODEL,
            "image":              image_path,
            "image_url":          image_url or "",
            "query":              args.query,
            "ocr_extracted_text": ocr_text,
            "image_answer":       img_result.get("text") or "",
            "image_latency_ms":   img_ms,
            "image_error":        img_result.get("error") or "",
            "ocr_answer":         ocr_result.get("text") or "",
            "ocr_latency_ms":     ocr_ms,
            "ocr_error":          ocr_result.get("error") or "",
            "faster_mode":        faster,
            "delta_ms":           delta,
            "verdict":            judgment["verdict"],
            "verdict_reason":     judgment["reason"],
        })
        print(f"  CSV row appended to: {args.csv}\n")


if __name__ == "__main__":
    main()
