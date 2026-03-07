#!/usr/bin/env python3
"""CLI tool for Braille image transcription."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import BraillePipeline


def main():
    parser = argparse.ArgumentParser(
        description='Transcribe Braille images to English text (Grade 1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python transcribe.py image.jpg
  python transcribe.py image.jpg --confidence 0.2
  python transcribe.py image.jpg --verbose
        '''
    )
    parser.add_argument('image', help='Path to the Braille image')
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.15,
        help='Detection confidence threshold (default: 0.15)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output including Braille Unicode and stats'
    )
    parser.add_argument(
        '--model', '-m',
        help='Path to YOLOv8 model weights (uses default if not specified)'
    )
    parser.add_argument(
        '--preprocess', '-p',
        action='store_true',
        help='Apply CLAHE preprocessing (helps with low contrast images)'
    )

    args = parser.parse_args()

    # Check image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Run pipeline
    pipeline = BraillePipeline(
        model_path=args.model,
        confidence=args.confidence,
        preprocess=args.preprocess
    )
    result = pipeline.transcribe(str(image_path))

    # Output
    if args.verbose:
        print("=== Braille Unicode ===")
        print(result['braille'])
        print()
        print("=== English Text ===")
        print(result['text'])
        print()
        print("=== Stats ===")
        stats = result['stats']
        print(f"Cells detected: {stats['cells']}")
        print(f"Lines: {stats['lines']}")
        print(f"Avg confidence: {stats['avg_confidence']:.2%}")
    else:
        print(result['text'])


if __name__ == '__main__':
    main()
