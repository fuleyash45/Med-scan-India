import re
from datetime import datetime

# ── All date patterns found on Indian medicine strips ──────
PATTERNS = [
    r'EXP(?:IRY)?[:\s\.]*(\d{2}[\/\-]\d{4})',
    r'EXP(?:IRY)?[:\s\.]*([A-Z]{3}[\-\s]\d{2,4})',
    r'USE\s*BEFORE[:\s]*(\d{2}[\/\-]\d{4})',
    r'USE\s*BEFORE[:\s]*([A-Z]{3}[\-\s]\d{2,4})',
    r'(\d{2}[\/]\d{4})',
    r'([A-Z]{3}[\-]\d{4})',
    r'([A-Z]{3}[\s]\d{4})',
]


def get_expiry(text):
    """
    Extracts expiry date string from raw OCR text.
    Returns date string like "12/2025" or "JAN-2026" or None.
    """
    if not text:
        return None

    text_upper = text.upper()

    for pattern in PATTERNS:
        match = re.search(pattern, text_upper)
        if match:
            return match.group(1).strip()

    return None


def parse_expiry_date(expiry_str):
    """
    Parses expiry string into datetime object.
    Handles: MM/YYYY, MMM-YYYY, MMM YYYY, MMM-YY
    """
    if not expiry_str:
        return None

    expiry_str = expiry_str.strip().upper()

    formats = ["%m/%Y", "%b-%Y", "%b %Y", "%b-%y", "%m-%Y"]

    for fmt in formats:
        try:
            return datetime.strptime(expiry_str, fmt)
        except ValueError:
            continue

    return None


def get_alert(expiry_str):
    """
    Returns alert level and human readable message.

    Returns tuple: (status, message)

    status values:
        green   → safe, more than 90 days left
        yellow  → caution, 30-90 days left
        red     → urgent, less than 30 days left
        expired → already expired
        unknown → could not detect or parse date
    """
    if not expiry_str:
        return "unknown", "Expiry date not detected — check strip manually"

    exp_date = parse_expiry_date(expiry_str)

    if not exp_date:
        return "unknown", f"Could not read expiry format: {expiry_str}"

    days_left = (exp_date - datetime.now()).days

    if days_left < 0:
        return (
            "expired",
            f"EXPIRED — {abs(days_left)} days ago. Do not use this medicine."
        )

    if days_left < 30:
        return (
            "red",
            f"Expires in {days_left} days — use immediately or replace"
        )

    if days_left < 90:
        return (
            "yellow",
            f"Expires in {days_left} days — monitor your stock"
        )

    return (
        "green",
        f"Expiry is safe — {days_left} days remaining"
    )