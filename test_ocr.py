from modules.ocr import extract_text

IMAGE_PATH = "test_strip.jpeg"

with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()

print("Running OCR...")
result = extract_text(image_bytes)

print("Source :", result["source"])
print("Error  :", result["error"])
print()
print("Match text:")
print(result["match_text"])
print()
print("Full OCR:")
print(result["raw_text"])