try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    raise ImportError(f"pip install pillow") from e


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
