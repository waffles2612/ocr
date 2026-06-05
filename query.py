import base64
import json
import os
import time
from pathlib import Path

try:
    import requests
except ImportError as e:
    raise ImportError("Missing dependency: requests. Run: pip install requests") from e

try:
    from config import API_KEY
except ImportError:
    API_KEY = None  # will be injected at runtime by app.py

try:
    from config import GITHUB_TOKEN
except ImportError:
    GITHUB_TOKEN = None  # will be injected at runtime by app.py

# ── Config ─────────────────────────────────────────────────────────────────────

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GITHUB_URL   = "https://models.inference.ai.azure.com/chat/completions"

# ── Helpers ────────────────────────────────────────────────────────────────────

def is_gemini(model: str) -> bool:
    return model.startswith("gemini")

def is_github(model: str) -> bool:
    return model.startswith("gpt-")


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


def image_to_data_url(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp", "bmp": "image/bmp"}.get(suffix.lstrip("."), "image/jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


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


def call_github(messages: list, model: str) -> dict:
    import query as _self
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_self.GITHUB_TOKEN}",
    }
    start = time.time()
    try:
        resp = requests.post(GITHUB_URL, headers=headers,
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
