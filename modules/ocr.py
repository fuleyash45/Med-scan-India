import requests
import base64
import os
import io
import re
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter

load_dotenv()


# ── Image Preprocessor ────────────────────────────────────
def preprocess_image(image_bytes):
    """
    Enhances image before sending to OCR.
    Fixes: low contrast, small size, blur.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # Convert transparent PNG to RGB
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Resize if too small
        w, h = img.size
        if w < 640 or h < 640:
            scale = max(640 / w, 640 / h)
            img = img.resize(
                (int(w * scale), int(h * scale)),
                Image.LANCZOS
            )

        # Sharpen text edges
        img = img.filter(ImageFilter.SHARPEN)

        # Boost contrast for faded strips
        img = ImageEnhance.Contrast(img).enhance(1.3)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    except Exception as e:
        print(f"[preprocess] Warning: {e} — using original")
        return image_bytes


# ── OCR.space API (Primary) ───────────────────────────────
def extract_text_ocrspace(image_bytes):
    """
    Primary OCR using OCR.space free API.
    25,000 free requests per month.
    No credit card needed.
    """
    api_key = os.getenv("OCR_SPACE_KEY")

    if not api_key or api_key == "paste_your_ocrspace_key_here":
        raise EnvironmentError("OCR_SPACE_KEY not set in .env file")

    # Encode image to base64
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    base64_image = f"data:image/jpeg;base64,{encoded}"

    payload = {
        "apikey": api_key,
        "base64Image": base64_image,
        "language": "eng",
        "isOverlayRequired": False,
        "detectOrientation": True,
        "isCreateSearchablePdf": False,
        "isSearchablePdfHideTextLayer": False,
        "scale": True,
        "isTable": False,
        "OCREngine": 2  # Engine 2 is more accurate for printed text
    }

    try:
        response = requests.post(
            "https://api.ocr.space/parse/image",
            data=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        # Check for API errors
        if result.get("IsErroredOnProcessing"):
            error_msg = result.get("ErrorMessage", ["Unknown error"])[0]
            raise ValueError(f"OCR.space error: {error_msg}")

        # Extract text from response
        parsed_results = result.get("ParsedResults", [])
        if not parsed_results:
            return ""

        text = parsed_results[0].get("ParsedText", "")
        return text.strip()

    except requests.exceptions.Timeout:
        raise TimeoutError("OCR.space timed out — check internet")

    except requests.exceptions.ConnectionError:
        raise ConnectionError("No internet — switching to offline OCR")

    except requests.exceptions.HTTPError as e:
        if response.status_code == 403:
            raise PermissionError("Invalid OCR.space API key")
        elif response.status_code == 429:
            raise RuntimeError("OCR.space daily limit reached")
        else:
            raise RuntimeError(f"HTTP error: {response.status_code}")


# ── EasyOCR (Offline Fallback) ────────────────────────────
def extract_text_easyocr(image_bytes):
    """
    Offline fallback using EasyOCR.
    Runs locally — no internet needed.
    Accuracy: 85-90% on clear images.
    First run downloads model (~100MB) automatically.
    """
    try:
        import easyocr
        import numpy as np

        # Load reader — English only for speed
        # gpu=False ensures it works on your 4GB RAM laptop
        reader = easyocr.Reader(["en"], gpu=False)

        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img)

        results = reader.readtext(img_array)

        # Join all detected text blocks
        text = "\n".join([res[1] for res in results])
        return text.strip()

    except ImportError:
        raise ImportError(
            "EasyOCR not installed. Run: pip install easyocr"
        )
    except Exception as e:
        raise RuntimeError(f"EasyOCR failed: {e}")


# ── Tesseract (Last Resort) ───────────────────────────────
def extract_text_tesseract(image_bytes):
    """
    Last resort OCR using Tesseract.
    Accuracy: 75-82%. Slowest option.
    """
    try:
        import pytesseract

        img = Image.open(io.BytesIO(image_bytes))
        config = r"--oem 3 --psm 6 -l eng"
        text = pytesseract.image_to_string(img, config=config)
        return text.strip()

    except ImportError:
        raise ImportError("pytesseract not installed")

    except Exception as e:
        raise RuntimeError(f"Tesseract failed: {e}")


# ── Text Cleaner ──────────────────────────────────────────
def clean_text(raw_text):
    """
    Removes OCR noise while keeping medicine text.
    """
    if not raw_text:
        return ""

    text = raw_text.replace("\x00", "").replace("\r", "\n")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\w)[|\\~`^]{1,2}(?!\w)", "", text)

    return text.strip()


# ── Medicine Line Extractor ───────────────────────────────
def extract_medicine_lines(raw_text):
    """
    Picks lines most likely to contain medicine name.
    Removes: addresses, phone numbers, long legal text.
    Returns top 5 candidate lines for fuzzy matching on a single line.
    """
    if not raw_text:
        return ""

    lines = raw_text.split("\n")
    candidates = []

    for line in lines:
        line = line.strip()

        # Skip empty or too short
        if len(line) < 3:
            continue

        # Skip pure numbers
        if re.match(r"^[\d\s\.\-\/]+$", line):
            continue

        # Skip address / contact lines
        if re.search(
            r"(plot|road|street|tel:|ph:|www\.|\.com|nagar|dist\.)",
            line.lower()
        ):
            continue

        # Skip long disclaimer text
        if len(line) > 60:
            continue

        candidates.append(line)

    # Join all candidates on a single line, removing any internal newlines
    match_text = " ".join(candidates[:5])
    # Remove any newlines that might have been in the text
    match_text = match_text.replace("\n", " ").replace("\r", " ")
    # Clean up multiple spaces
    match_text = re.sub(r" {2,}", " ", match_text)
    return match_text.strip()


# ── MAIN FUNCTION — call this from app.py ─────────────────
def extract_text(image_bytes, use_api=True):
    """
    Main OCR function. Tries 3 methods in order:
    1. OCR.space API (best, free, needs internet)
    2. EasyOCR (good, free, offline)
    3. Tesseract (basic, free, offline)

    Returns dict:
        raw_text   — full OCR output
        clean_text — noise removed
        match_text — best lines for medicine matching
        source     — which OCR was used
        error      — error message or None
    """
    result = {
        "raw_text": "",
        "clean_text": "",
        "match_text": "",
        "source": "failed",
        "error": None
    }

    # Preprocess image
    try:
        processed = preprocess_image(image_bytes)
    except Exception:
        processed = image_bytes

    # Method 1 — OCR.space API
    if use_api:
        try:
            raw = extract_text_ocrspace(processed)
            if raw:
                result["raw_text"] = raw
                result["clean_text"] = clean_text(raw)
                result["match_text"] = extract_medicine_lines(raw)
                result["source"] = "ocr.space"
                return result

        except EnvironmentError:
            pass  # No key — try EasyOCR

        except (TimeoutError, ConnectionError):
            result["error"] = "No internet — using offline OCR"

        except (PermissionError, RuntimeError, ValueError) as e:
            result["error"] = str(e)

    # Method 2 — EasyOCR offline
    try:
        raw = extract_text_easyocr(processed)
        if raw:
            result["raw_text"] = raw
            result["clean_text"] = clean_text(raw)
            result["match_text"] = extract_medicine_lines(raw)
            result["source"] = "easyocr"
            return result

    except Exception as e:
        result["error"] = f"EasyOCR failed: {e}"

    # Method 3 — Tesseract last resort
    try:
        raw = extract_text_tesseract(processed)
        result["raw_text"] = raw
        result["clean_text"] = clean_text(raw)
        result["match_text"] = extract_medicine_lines(raw)
        result["source"] = "tesseract"
        return result

    except Exception as e:
        result["error"] = f"All OCR methods failed: {str(e)}"
        return result