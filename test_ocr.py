#!/usr/bin/env python3
"""
Simple test script for medicine strip OCR
Usage: python test_ocr.py
"""

import os
from modules.ocr import extract_text

# Use the actual filename that exists
TEST_IMAGE = 'test_strip.jpg.jpeg'

def main():
    print("=" * 60)
    print("MEDICINE STRIP OCR TEST")
    print("=" * 60)
    
    # Check if image exists
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ Error: Image file '{TEST_IMAGE}' not found!")
        print(f"   Available image files: {[f for f in os.listdir('.') if f.endswith(('.jpg', '.jpeg', '.png'))]}")
        return
    
    print(f"✓ Image found: {TEST_IMAGE}")
    print(f"  File size: {os.path.getsize(TEST_IMAGE)} bytes\n")
    
    # Read image
    try:
        with open(TEST_IMAGE, 'rb') as f:
            image_data = f.read()
        print("✓ Image loaded successfully\n")
    except Exception as e:
        print(f"❌ Error reading image: {e}")
        return
    
    # Extract text using OCR (use_api=False will skip OCR.space and use EasyOCR)
    print("Running OCR (using EasyOCR offline)...")
    print("This may take a moment on first run...\n")
    
    try:
        result = extract_text(image_data, use_api=False)
    except Exception as e:
        print(f"❌ Error in extract_text: {e}")
        return
    
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"Source: {result['source']}")
    print(f"Error: {result['error']}")
    print()
    
    print("--- Raw OCR Text ---")
    print(result['raw_text'][:500] if result['raw_text'] else "(empty)")
    print()
    
    print("--- Cleaned Text ---")
    print(result['clean_text'][:500] if result['clean_text'] else "(empty)")
    print()
    
    print("--- Medicine Text (for matching) ---")
    print(result['match_text'] if result['match_text'] else "(empty)")
    print()
    
    if result['error']:
        print(f"⚠️  Error occurred: {result['error']}")
    else:
        print("✓ OCR completed successfully!")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
