"""End-to-end Braille OCR pipeline: Image → English text."""

import os

from .detector import BrailleDetector
from .interpreter import interpret_lines
from .preprocess import preprocess_for_detection


class BraillePipeline:
    """End-to-end pipeline for Braille image transcription."""

    def __init__(self, model_path: str = None, confidence: float = 0.15, preprocess: bool = False):
        """Initialize the pipeline.

        Args:
            model_path: Path to YOLOv8 model weights. Uses default if None.
            confidence: Minimum confidence threshold for detection.
            preprocess: Whether to apply CLAHE preprocessing.
        """
        self.detector = BrailleDetector(model_path, confidence)
        self.preprocess = preprocess

    def transcribe(self, image_path: str) -> dict:
        """Transcribe a Braille image to English text.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict with:
                - text: English transcription
                - braille: Braille Unicode text
                - lines: Detailed detection data
                - stats: Detection statistics
        """
        # Optional preprocessing
        temp_path = None
        if self.preprocess:
            temp_path = preprocess_for_detection(image_path)
            image_path = temp_path

        try:
            # Stage 1: Detect cells
            lines = self.detector.detect(image_path)
        finally:
            # Cleanup temp file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

        if not lines:
            return {
                'text': '',
                'braille': '',
                'lines': [],
                'stats': {'cells': 0, 'lines': 0, 'avg_confidence': 0.0}
            }

        # Get Braille Unicode strings
        braille_lines = self.detector.get_braille_unicode(lines)

        # Stage 2: Interpret to English
        english_text = interpret_lines(braille_lines)

        # Calculate stats
        all_confidences = [cell['confidence'] for line in lines for cell in line]
        total_cells = len(all_confidences)
        avg_confidence = sum(all_confidences) / total_cells if total_cells > 0 else 0.0

        return {
            'text': english_text,
            'braille': '\n'.join(braille_lines),
            'lines': lines,
            'stats': {
                'cells': total_cells,
                'lines': len(lines),
                'avg_confidence': avg_confidence,
            }
        }
