"""
Check which free vision models are available on your OpenRouter account.
Usage: python check_models.py
"""

import sys
import requests

sys.stdout.reconfigure(encoding="utf-8")

try:
    from config import API_KEY
except ImportError:
    print("[error] config.py not found. Create it with: API_KEY = 'sk-or-v1-...'")
    sys.exit(1)

print("\n  Fetching models from OpenRouter...\n")

resp = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {API_KEY}"},
    timeout=15,
)
data = resp.json()

free_vision = []
free_text   = []

for model in data.get("data", []):
    model_id = model["id"]
    if ":free" not in model_id:
        continue
    modalities = model.get("architecture", {}).get("input_modalities", [])
    if "image" in modalities:
        free_vision.append(model_id)
    else:
        free_text.append(model_id)

print("=" * 60)
print("  FREE VISION MODELS (support image input)")
print("=" * 60)
if free_vision:
    for m in free_vision:
        print(f"  ✅ {m}")
else:
    print("  (none found)")

print()
print("=" * 60)
print("  FREE TEXT-ONLY MODELS (OCR mode only)")
print("=" * 60)
if free_text:
    for m in free_text:
        print(f"  📝 {m}")
else:
    print("  (none found)")

print()