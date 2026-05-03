import os
import io
import re
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from dotenv import load_dotenv

load_dotenv()

_reader = None


def get_reader():
    global _reader
    if _reader is None:
        import easyocr
        print("Loading EasyOCR model...")
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("EasyOCR ready")
    return _reader


def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # Auto rotate from phone EXIF
        try:
            from PIL import ExifTags
            exif = img._getexif()
            if exif:
                for tag, value in exif.items():
                    if ExifTags.TAGS.get(tag) == "Orientation":
                        if value == 3:
                            img = img.rotate(180, expand=True)
                        elif value == 6:
                            img = img.rotate(270, expand=True)
                        elif value == 8:
                            img = img.rotate(90, expand=True)
                        break
        except Exception:
            pass

        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        w, h = img.size
        if w < 640 or h < 640:
            scale = max(640 / w, 640 / h)
            img = img.resize(
                (int(w * scale), int(h * scale)),
                Image.LANCZOS
            )

        img = img.filter(ImageFilter.SHARPEN)
        img = ImageEnhance.Contrast(img).enhance(1.4)
        img = ImageEnhance.Brightness(img).enhance(1.1)

        return img

    except Exception as e:
        print(f"Preprocess warning: {e}")
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def clean_text(raw_text):
    if not raw_text:
        return ""
    text = raw_text.replace("\x00", "").replace("\r", "\n")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\w)[|\\~`^]{1,2}(?!\w)", "", text)
    return text.strip()


def extract_medicine_lines(raw_text):
    if not raw_text:
        return ""
    lines = raw_text.split("\n")
    candidates = []
    for line in lines:
        line = line.strip()
        if len(line) < 3:
            continue
        if re.match(r"^[\d\s\.\-\/]+$", line):
            continue
        if re.search(
            r"(plot|road|street|tel:|ph:|www\.|\.com|nagar|dist\.|gst|fax)",
            line.lower()
        ):
            continue
        if len(line) > 60:
            continue
        if re.match(r"^[^a-zA-Z0-9]+$", line):
            continue
        candidates.append(line)
    return " ".join(candidates[:5])


def extract_text(image_bytes):
    """
    Main OCR function. Call this from app.py.
    Always returns a dict with these keys:
        raw_text   → full OCR output
        clean_text → noise removed
        match_text → best lines for medicine matching
        source     → which OCR engine was used
        error      → error message or None
    """
    result = {
        "raw_text": "",
        "clean_text": "",
        "match_text": "",
        "source": "failed",
        "error": None
    }

    # Preprocess
    try:
        img = preprocess_image(image_bytes)
    except Exception as e:
        result["error"] = f"Image load failed: {str(e)}"
        return result

    # Method 1 — EasyOCR
    try:
        reader = get_reader()
        img_array = np.array(img)
        results = reader.readtext(img_array, detail=1, paragraph=False)

        if results:
            results_sorted = sorted(results, key=lambda x: x[0][0][1])
            lines = [res[1] for res in results_sorted]
            raw = "\n".join(lines)

            result["raw_text"] = raw
            result["clean_text"] = clean_text(raw)
            result["match_text"] = extract_medicine_lines(raw)
            result["source"] = "easyocr"
            return result
        else:
            result["error"] = "No text detected — try clearer photo"
            return result

    except Exception as e:
        result["error"] = f"EasyOCR failed: {str(e)}"

    # Method 2 — Tesseract fallback
    try:
        import pytesseract
        config = r"--oem 3 --psm 6 -l eng"
        raw = pytesseract.image_to_string(img, config=config).strip()

        result["raw_text"] = raw
        result["clean_text"] = clean_text(raw)
        result["match_text"] = extract_medicine_lines(raw)
        result["source"] = "tesseract"
        return result

    except Exception as e:
        result["error"] = f"All OCR failed: {str(e)}"
        return result