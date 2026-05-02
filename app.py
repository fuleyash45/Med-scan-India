"""
Med-scan India - Medical Strip Authentication App
Main application entry point
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from modules import (
    OCRProcessor,
    StripMatcher,
    ExpiryChecker,
    FakeDetector
)


def print_banner():
    """Print application banner"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║              Med-scan India - Medical Strip                ║
║                   Authentication App                       ║
╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def main_menu():
    """Display main menu and handle user choices"""
    while True:
        print("\n" + "="*50)
        print(" MAIN MENU")
        print("="*50)
        print("1. Process a medical strip image")
        print("2. Check strip expiry")
        print("3. Detect fake strips")
        print("4. Compare two strips")
        print("5. Exit")
        print("="*50)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            process_image()
        elif choice == '2':
            check_expiry()
        elif choice == '3':
            detect_fake()
        elif choice == '4':
            compare_strips()
        elif choice == '5':
            print("\nThank you for using Med-scan India!")
            break
        else:
            print("\nInvalid choice. Please try again.")


def process_image():
    """Process a medical strip image"""
    print("\n--- Process Medical Strip ---")
    image_path = input("Enter image path: ").strip()
    
    if not os.path.exists(image_path):
        print(f"Error: File not found - {image_path}")
        return
    
    try:
        ocr = OCRProcessor()
        result = ocr.extract_all_info(image_path)
        
        print("\n--- Results ---")
        print(f"Raw Text:\n{result['raw_text']}")
        print(f"\nDates: {result['dates']}")
        print(f"Batch Number: {result['batch_number']}")
    except Exception as e:
        print(f"Error: {e}")


def check_expiry():
    """Check strip expiry date"""
    print("\n--- Check Expiry ---")
    image_path = input("Enter image path: ").strip()
    
    if not os.path.exists(image_path):
        print(f"Error: File not found - {image_path}")
        return
    
    try:
        ocr = OCRProcessor()
        ocr_result = ocr.extract_all_info(image_path)
        
        expiry = ExpiryChecker()
        result = expiry.check_from_ocr_data(ocr_result)
        
        print("\n--- Expiry Status ---")
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Expiry Date: {result['expiry_date']}")
            print(f"Status: {result['status']}")
            print(f"Days Left: {result['days_left']}")
            print(f"Is Valid: {result['is_valid']}")
    except Exception as e:
        print(f"Error: {e}")


def detect_fake():
    """Detect fake strips"""
    print("\n--- Fake Detection ---")
    print("Note: Model must be trained first with genuine and fake samples.")
    
    image_path = input("Enter image path: ").strip()
    
    if not os.path.exists(image_path):
        print(f"Error: File not found - {image_path}")
        return
    
    try:
        detector = FakeDetector()
        print("\nNote: Please train the model first using genuine and fake sample folders.")
        print("Use: detector.train('data/genuine', 'data/fake')")
    except Exception as e:
        print(f"Error: {e}")


def compare_strips():
    """Compare two strip images"""
    print("\n--- Compare Strips ---")
    image1 = input("Enter first image path: ").strip()
    image2 = input("Enter second image path: ").strip()
    
    if not os.path.exists(image1):
        print(f"Error: File not found - {image1}")
        return
    
    if not os.path.exists(image2):
        print(f"Error: File not found - {image2}")
        return
    
    try:
        matcher = StripMatcher()
        similarity = matcher.calculate_similarity(image1, image2)
        is_match = matcher.is_match(image1, image2)
        
        print("\n--- Comparison Results ---")
        print(f"Similarity: {similarity:.2%}")
        print(f"Match: {is_match}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main entry point"""
    print_banner()
    
    # Check if required packages are available
    try:
        import pytesseract
        from PIL import Image
        import numpy as np
        print("✓ All required packages installed")
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("\nPlease install required packages:")
        print("pip install pytesseract Pillow numpy scikit-learn scikit-image")
        return
    
    # Run main menu
    main_menu()


if __name__ == "__main__":
    main()