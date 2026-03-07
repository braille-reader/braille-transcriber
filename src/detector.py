"""Braille cell detector using YOLOv8."""

import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO


# Default paths
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "DotNeuralNet" / "weights" / "yolov8_braille.pt"
BRAILLE_MAP_PATH = Path(__file__).parent.parent / "DotNeuralNet" / "src" / "utils" / "braille_map.json"

# Load braille map (dot pattern → unicode)
with open(BRAILLE_MAP_PATH) as f:
    DOT_TO_UNICODE = json.load(f)


class BrailleDetector:
    """Detects Braille cells in images using YOLOv8."""

    def __init__(self, model_path: str = None, confidence: float = 0.15):
        """Initialize the detector.

        Args:
            model_path: Path to YOLOv8 model weights. Uses default if None.
            confidence: Minimum confidence threshold for detection.
        """
        if model_path is None:
            model_path = str(DEFAULT_MODEL_PATH)

        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, image_path: str) -> list[list[dict]]:
        """Detect Braille cells in an image.

        Args:
            image_path: Path to the image file.

        Returns:
            List of lines, where each line is a list of cell dicts with:
                - x, y: center position
                - dots: 6-digit dot pattern string
                - unicode: Braille Unicode character
                - confidence: detection confidence
        """
        results = self.model.predict(image_path, conf=self.confidence, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return []

        lines = self._parse_boxes(boxes)
        return lines

    def _parse_boxes(self, boxes) -> list[list[dict]]:
        """Parse YOLO boxes into sorted lines of cells."""
        # Extract to numpy array
        n = len(boxes)
        data = np.zeros((n, 6))
        data[:, 0] = boxes.xywh[:, 0].cpu().numpy()  # x center
        data[:, 1] = boxes.xywh[:, 1].cpu().numpy()  # y center
        data[:, 2] = boxes.xywh[:, 2].cpu().numpy()  # width
        data[:, 3] = boxes.xywh[:, 3].cpu().numpy()  # height
        data[:, 4] = boxes.conf.cpu().numpy()        # confidence
        data[:, 5] = boxes.cls.cpu().numpy()         # class index

        # Sort by Y coordinate
        data = data[data[:, 1].argsort()]

        # Find line breaks (Y gaps larger than half average height)
        avg_height = np.mean(data[:, 3])
        y_threshold = avg_height / 2
        y_diffs = np.diff(data[:, 1])
        break_indices = np.where(y_diffs > y_threshold)[0]

        # Split into lines
        raw_lines = np.split(data, break_indices + 1)

        # Convert to cell dicts, sorted by X within each line
        lines = []
        for raw_line in raw_lines:
            # Sort by X
            raw_line = raw_line[raw_line[:, 0].argsort()]

            cells = []
            for row in raw_line:
                class_idx = int(row[5])
                dots = self.model.names[class_idx]
                unicode_char = DOT_TO_UNICODE.get(dots, '?')

                cells.append({
                    'x': row[0],
                    'y': row[1],
                    'dots': dots,
                    'unicode': unicode_char,
                    'confidence': row[4],
                })
            lines.append(cells)

        return lines

    def get_braille_unicode(self, lines: list[list[dict]]) -> list[str]:
        """Extract Braille Unicode strings from detected lines.

        Args:
            lines: Output from detect()

        Returns:
            List of Braille Unicode strings, one per line
        """
        return [''.join(cell['unicode'] for cell in line) for line in lines]
