"""
Streamlit UI — OCR vs Image LLM Comparison Tool
"""

import csv
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

# ── Tesseract path (Windows local; ignored on Linux/Streamlit Cloud) ──────────
_tess_dir = r"C:\Program Files\Tesseract-OCR"
if sys.platform == "win32" and _tess_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _tess_dir + os.pathsep + os.environ.get("PATH", "")

try:
    import pytesseract
    from PIL import Image
    import requests
    if sys.platform == "win32":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError as e:
    st.error(f"Missing dependency: {e}. Run: pip install pytesseract pillow requests")
    st.stop()

from ocr import run_ocr
from visualize_ocr import draw_boxes
from query import call_groq, call_gemini, is_gemini, judge_answers, upload_image, VISION_MODEL, JUDGE_MODEL

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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCR vs Image LLM Tester",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 OCR vs Image LLM Comparison")
st.caption("Upload an image, run OCR, then ask questions — compare answers from raw image vs OCR text.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    model = st.selectbox("Vision model", [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ])

    if is_gemini(model):
        api_key = None
        gemini_key = get_gemini_key()
        if not gemini_key:
            gemini_key = st.text_input("Gemini API Key", type="password",
                                       placeholder="AIza...")
            if not gemini_key:
                st.warning("Enter your Gemini API key to continue.")
    else:
        gemini_key = None
        api_key = get_api_key()
        if not api_key:
            api_key = st.text_input("Groq API Key", type="password",
                                    placeholder="gsk_...")
            if not api_key:
                st.warning("Enter your Groq API key to continue.")

    lang = st.selectbox("OCR language", ["eng", "hin", "fra", "deu", "spa"], index=0)
    min_conf = st.slider("Min bbox confidence", 0, 100, 40,
                         help="Hide OCR words below this confidence from the annotated image")

    st.divider()
    st.caption("Color guide for bboxes:")
    st.markdown("🟢 High conf (≥ 80%)  \n🟠 Medium (50–79%)  \n🔴 Low (< 50%)")

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

    if st.button("▶ Run OCR", type="primary"):
        with st.spinner("Running Tesseract..."):
            ocr_data = run_ocr(tmp_path, lang=lang)
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

# ── Step 2: Query ─────────────────────────────────────────────────────────────
if st.session_state.ocr_data:
    st.divider()
    st.header("Step 2 — Ask a Question")

    query = st.text_input("Your question", placeholder="What is the total amount?")

    active_key = gemini_key if is_gemini(model) else api_key
    if st.button("▶ Run Query", type="primary", disabled=not (query and active_key)):
        ocr_data = st.session_state.ocr_data
        tmp_path  = ocr_data["_tmp_path"]
        ocr_text  = ocr_data["plain_text"]

        if is_gemini(model):
            with st.spinner("Querying with image..."):
                from PIL import Image as PILImage
                pil_img = PILImage.open(tmp_path)
                img_result = call_gemini([pil_img, query], model, gemini_key)
                image_url = None

            with st.spinner("Querying with OCR text..."):
                ocr_prompt = (
                    f"The following text was extracted via OCR from an image:\n\n"
                    f"{ocr_text}\n\nUsing only this text, answer:\n{query}"
                )
                ocr_result = call_gemini(ocr_prompt, model, gemini_key)

        else:
            import query as qmod
            qmod.API_KEY = api_key

            with st.spinner("Uploading image..."):
                image_url = None
                try:
                    image_url = upload_image(tmp_path)
                except Exception as e:
                    st.warning(f"Image upload failed ({e}) — image mode will be skipped.")

            with st.spinner("Querying with image..."):
                if image_url:
                    img_messages = [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": query},
                    ]}]
                    img_result = call_groq(img_messages, model)
                else:
                    img_result = {"text": None, "latency_ms": 0, "error": "Skipped (upload failed)"}

            with st.spinner("Querying with OCR text..."):
                ocr_messages = [{"role": "user", "content": (
                    f"The following text was extracted via OCR from an image:\n\n"
                    f"{ocr_text}\n\nUsing only this text, answer:\n{query}"
                )}]
                ocr_result = call_groq(ocr_messages, model)

        with st.spinner("Judging answers..."):
            if img_result.get("error") or ocr_result.get("error"):
                judgment = {"verdict": "SKIP", "reason": "One or both answers errored out"}
            else:
                judgment = judge_answers(query, img_result["text"], ocr_result["text"],
                                         gemini_key=gemini_key)

        img_ms = img_result["latency_ms"]
        ocr_ms = ocr_result["latency_ms"]
        faster = "Image" if img_ms < ocr_ms else "OCR text"

        row = {
            "timestamp":          time.strftime("%Y-%m-%d %H:%M:%S"),
            "model":              model,
            "image":              ocr_data["image"],
            "image_url":          image_url or "",
            "query":              query,
            "ocr_extracted_text": ocr_text,
            "image_answer":       img_result.get("text") or "",
            "image_latency_ms":   img_ms,
            "image_error":        img_result.get("error") or "",
            "ocr_answer":         ocr_result.get("text") or "",
            "ocr_latency_ms":     ocr_ms,
            "ocr_error":          ocr_result.get("error") or "",
            "faster_mode":        faster,
            "delta_ms":           abs(img_ms - ocr_ms),
            "verdict":            judgment["verdict"],
            "verdict_reason":     judgment["reason"],
        }

        if st.session_state.results is None:
            st.session_state.results = []
        st.session_state.results.append(row)

    # ── Display latest result ─────────────────────────────────────────────────
    if st.session_state.results:
        row = st.session_state.results[-1]

        st.subheader("Results")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**IMAGE mode**")
            if row["image_error"]:
                st.error(row["image_error"])
            else:
                st.success(row["image_answer"])
            st.caption(f"Latency: {row['image_latency_ms']:,} ms")

        with c2:
            st.markdown("**OCR TEXT mode**")
            if row["ocr_error"]:
                st.error(row["ocr_error"])
            else:
                st.info(row["ocr_answer"])
            st.caption(f"Latency: {row['ocr_latency_ms']:,} ms")

        verdict = row["verdict"]
        if verdict == "MATCH":
            st.success(f"✅ MATCH — {row['verdict_reason']}")
        elif verdict == "MISMATCH":
            st.error(f"❌ MISMATCH — {row['verdict_reason']}")
        else:
            st.warning(f"⚠️ {verdict} — {row['verdict_reason']}")

# ── Step 3: Download CSV ──────────────────────────────────────────────────────
if st.session_state.results:
    st.divider()
    st.header("Step 3 — Download Results")

    fieldnames = [
        "timestamp", "model", "image", "image_url", "query",
        "ocr_extracted_text",
        "image_answer", "image_latency_ms", "image_error",
        "ocr_answer", "ocr_latency_ms", "ocr_error",
        "faster_mode", "delta_ms", "verdict", "verdict_reason",
    ]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(st.session_state.results)

    st.download_button(
        "⬇ Download CSV",
        data=buf.getvalue().encode("utf-8"),
        file_name="results.csv",
        mime="text/csv",
    )

    st.dataframe(st.session_state.results, use_container_width=True)
