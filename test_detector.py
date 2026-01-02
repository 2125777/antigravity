import cv2
import detector
import numpy as np

# Create a blank image with some text to see if EasyOCR picks it up
# This is a basic sanity check for the detector logic
def test_detector():
    # Create a white image
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    # Add some text simulating a license plate (though it won't be in a 'car' box)
    # So we'll mock the 'car' detection part in this test or just test the OCR reader
    
    det = detector.RIPASDetector()
    print("Detector initialized.")
    
    # Test OCR reader directly
    results = det.reader.readtext(img)
    print(f"OCR results on blank image: {results}")
    
    print("Test finished.")

if __name__ == "__main__":
    test_detector()
