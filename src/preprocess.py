"""Image preprocessing for Braille detection."""

import cv2
import numpy as np


def preprocess_image(image_path: str, output_path: str = None) -> np.ndarray:
    """Apply preprocessing to improve Braille detection.

    Steps:
    1. Convert to grayscale
    2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    3. Apply slight Gaussian blur to reduce noise

    Args:
        image_path: Path to input image
        output_path: Optional path to save preprocessed image

    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Convert back to BGR for YOLO (expects 3 channels)
    result = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    if output_path:
        cv2.imwrite(output_path, result)

    return result


def preprocess_for_detection(image_path: str) -> str:
    """Preprocess image and save to temp file for detection.

    Args:
        image_path: Path to input image

    Returns:
        Path to preprocessed temp image
    """
    import tempfile
    from pathlib import Path

    # Create temp file with same extension
    suffix = Path(image_path).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        temp_path = f.name

    preprocess_image(image_path, temp_path)
    return temp_path
