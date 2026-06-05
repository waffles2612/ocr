"""
Streamlit UI — OCR vs Image LLM Comparison Tool
"""

import json
import os
import re
import sys
import tempfile
from pathlib import Path

import streamlit as st


# ── Tesseract path (Windows local; ignored on Linux/Streamlit Cloud) ──────────
_tess_dir = r"C:\Program Files\Tesseract-OCR"
if sys.platform == "win32" and _tess_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _tess_dir + os.pathsep + os.environ.get("PATH", "")

try:
    import pytesseract
except ImportError:
    st.error("Missing dependency: pytesseract. Run: pip install pytesseract pillow")
    st.stop()

from ocr import run_ocr
from visualize_ocr import draw_boxes

from query import call_groq, call_gemini, call_github, is_gemini, is_github, image_to_data_url

# ── API keys (Streamlit secrets → config.py fallback) ────────────────────────
def get_api_key():
    try:
        return st.secrets["API_KEY"]
    except Exception:
        try:
            from config import API_KEY
            return API_KEY
        except ImportError:
            return None

def get_gemini_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        try:
            from config import GEMINI_API_KEY
            return GEMINI_API_KEY
        except (ImportError, NameError):
            return None

def get_github_token():
    try:
        return st.secrets["GITHUB_TOKEN"]
    except Exception:
        try:
            from config import GITHUB_TOKEN
            return GITHUB_TOKEN
        except (ImportError, NameError):
            return None

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCR vs Image Tester",
    layout="wide",
)

st.title("OCR vs Image LLM Comparison")
st.caption("Upload an image, run OCR, then ask questions ")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    model = st.selectbox("Vision model", [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        "gpt-4o-mini",
    ])

    if is_gemini(model):
        api_key = None
        github_token = None
        gemini_key = get_gemini_key()
        if not gemini_key:
            gemini_key = st.text_input("Gemini API Key", type="password",
                                       placeholder="AIza...")
            if not gemini_key:
                st.warning("Enter your Gemini API key to continue.")
    elif is_github(model):
        api_key = None
        gemini_key = None
        github_token = get_github_token()
        if not github_token:
            github_token = st.text_input("GitHub Token", type="password",
                                         placeholder="ghp_...")
            if not github_token:
                st.warning("Enter your GitHub token to continue.")
    else:
        gemini_key = None
        github_token = None
        api_key = get_api_key()
        if not api_key:
            api_key = st.text_input("Groq API Key", type="password",
                                    placeholder="gsk_...")
            if not api_key:
                st.warning("Enter your Groq API key to continue.")

    min_conf = 0

# ── Session state ─────────────────────────────────────────────────────────────
for key in ("ocr_data", "annotated_img", "results"):
    if key not in st.session_state:
        st.session_state[key] = None


# ── Step 1: Upload & OCR ──────────────────────────────────────────────────────
st.header("Step 1 — Upload Image & Run OCR")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original image")
        st.image(tmp_path, use_container_width=True)

    if st.button("Run OCR", type="primary"):
        with st.spinner("Running Tesseract..."):
            ocr_data = run_ocr(tmp_path)
            ocr_data["_tmp_path"] = tmp_path
            st.session_state.ocr_data = ocr_data
            ann_img, _ = draw_boxes(tmp_path, ocr_data["words"], min_conf=min_conf)
            st.session_state.annotated_img = ann_img
            st.session_state.results = []

    if st.session_state.ocr_data:
        ocr_data = st.session_state.ocr_data
        with col2:
            st.subheader("Annotated bboxes")
            st.image(st.session_state.annotated_img, use_container_width=True)

        with st.expander("Extracted text", expanded=False):
            st.text(ocr_data["plain_text"] or "(no text extracted)")

        st.success(f"OCR done — {ocr_data['word_count']} words extracted.")

def parse_llm_json(text: str) -> dict:
    """Extract {answer, quote} from an LLM response that should be JSON."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip("`").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {"answer": text, "quote": None}



def _ocr_prompt(ocr_text: str, context: str, bbox_context: str, query: str) -> str:
    return (
        f"The following text was extracted via OCR from an image:\n\n"
        f"{ocr_text}\n\n"
        f"Nearby words around \"{context}\" (with positions):\n{bbox_context}\n\n"
        f"Using only this text, answer the question below.\n"
        f"Respond ONLY with a JSON object — no extra text:\n"
        f'{{"answer": "your answer", "bbox": {{"x": <left>, "y": <top>, "width": <w>, "height": <h>}} }}\n'
        f"bbox must be the bounding box of the word(s) in the position data above that contain the answer "
        f"(compute the union rectangle if multiple words). Set bbox to null if the answer is not in the position data.\n\n"
        f"Question: {query}"
    )


def get_neighboring_words(words: list, context: str, radius: int = 150) -> list:
    context_lower = context.lower()
    anchors = [w for w in words if context_lower in w["text"].lower()]
    if not anchors:
        return words  # fallback: return all words if context not found
    cx = sum(w["left"] + w["width"] // 2 for w in anchors) / len(anchors)
    cy = sum(w["top"]  + w["height"] // 2 for w in anchors) / len(anchors)
    return [
        w for w in words
        if abs((w["left"] + w["width"] // 2) - cx) <= radius
        and abs((w["top"]  + w["height"] // 2) - cy) <= radius
    ]


# ── Step 2: Query ─────────────────────────────────────────────────────────────
if st.session_state.ocr_data:
    st.divider()
    st.header("Step 2 — Ask a Question")

    query   = st.text_input("Your question", placeholder="What is the total amount?")
    context = st.text_input("Context", placeholder="e.g. Total, Invoice No, Date")

    active_key = gemini_key if is_gemini(model) else (github_token if is_github(model) else api_key)
    if st.button("Run Query", type="primary", disabled=not (query and context.strip() and active_key)):
        ocr_data = st.session_state.ocr_data
        tmp_path  = ocr_data["_tmp_path"]
        ocr_text  = ocr_data["plain_text"]

        neighbors = get_neighboring_words(ocr_data["words"], context)
        bbox_lines = [
            f"[block {w['block_num']}, line {w['line_num']}] \"{w['text']}\" "
            f"at ({w['left']}, {w['top']}) size {w['width']}x{w['height']}"
            for w in neighbors
        ]
        bbox_context = "\n".join(bbox_lines)

        if is_gemini(model):
            with st.spinner("Querying with image..."):
                from PIL import Image as PILImage
                pil_img = PILImage.open(tmp_path)
                img_result = call_gemini([pil_img, query], model, gemini_key)

            with st.spinner("Querying with OCR text..."):
                ocr_result = call_gemini(_ocr_prompt(ocr_text, context, bbox_context, query), model, gemini_key)

        elif is_github(model):
            import query as qmod
            qmod.GITHUB_TOKEN = github_token

            with st.spinner("Querying with image..."):
                data_url = image_to_data_url(tmp_path)
                img_messages = [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": query},
                ]}]
                img_result = call_github(img_messages, model)

            with st.spinner("Querying with OCR text..."):
                ocr_messages = [{"role": "user", "content": _ocr_prompt(ocr_text, context, bbox_context, query)}]
                ocr_result = call_github(ocr_messages, model)

        else:
            import query as qmod
            qmod.API_KEY = api_key

            with st.spinner("Querying with image..."):
                data_url = image_to_data_url(tmp_path)
                img_messages = [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": query},
                ]}]
                img_result = call_groq(img_messages, model)

            with st.spinner("Querying with OCR text..."):
                ocr_messages = [{"role": "user", "content": _ocr_prompt(ocr_text, context, bbox_context, query)}]
                ocr_result = call_groq(ocr_messages, model)

        img_ms = img_result["latency_ms"]
        ocr_ms = ocr_result["latency_ms"]

        img_answer = img_result.get("text") or ""

        # OCR mode: LLM has the coordinates in context and returns bbox directly
        ocr_parsed  = parse_llm_json(ocr_result.get("text") or "")
        ocr_answer  = ocr_parsed.get("answer") or ocr_result.get("text") or ""
        raw_ocr_bbox = ocr_parsed.get("bbox") if not ocr_result.get("error") else None
        ocr_bbox = raw_ocr_bbox if (
            isinstance(raw_ocr_bbox, dict)
            and all(k in raw_ocr_bbox for k in ("x", "y", "width", "height"))
        ) else None

        st.session_state.results = {
            "img_result": img_result,
            "ocr_result": ocr_result,
            "img_answer": img_answer,
            "ocr_answer": ocr_answer,
            "ocr_bbox":   ocr_bbox,
            "img_ms":     img_ms,
            "ocr_ms":     ocr_ms,
        }

    # ── Display latest result ─────────────────────────────────────────────────
    if st.session_state.results:
        r = st.session_state.results

        st.subheader("Results")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**IMAGE mode**")
            if r["img_result"]["error"]:
                st.error(r["img_result"]["error"])
            else:
                st.success(r["img_answer"])
            st.caption(f"Latency: {r['img_ms']:,} ms")

        with c2:
            st.markdown("**OCR TEXT mode**")
            if r["ocr_result"]["error"]:
                st.error(r["ocr_result"]["error"])
            else:
                st.info(r["ocr_answer"])
            st.caption(f"Latency: {r['ocr_ms']:,} ms")

        # ── Answer location (OCR mode only) ──────────────────────────────────
        ocr_bbox = r.get("ocr_bbox")
        if ocr_bbox:
            b = ocr_bbox
            st.caption(f"bbox — x:{b['x']} y:{b['y']} w:{b['width']} h:{b['height']}")

