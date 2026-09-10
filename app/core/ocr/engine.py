from typing import Dict, List

import numpy as np
import pytesseract

from app.core.preprocessing.text import TextPreprocessor


class OCREngine:
    """Tesseract-based OCR engine with line grouping and confidence filtering."""

    def __init__(self, lang: str = "eng", min_conf: int = 35):
        self.lang = lang
        self.min_conf = min_conf

    def extract_lines(self, image: np.ndarray) -> List[Dict]:
        """Extract OCR lines grouped by block/paragraph/line.

        Groups words into lines, filters by confidence, and returns
        structured line data with bounding boxes.

        Args:
            image: Preprocessed binary image.

        Returns:
            Sorted list of line dictionaries with text, bbox, and confidence.
        """
        data = pytesseract.image_to_data(
            image, lang=self.lang, output_type=pytesseract.Output.DICT
        )
        grouped: Dict[tuple, List[Dict]] = {}

        for i in range(len(data["text"])):
            word = (data["text"][i] or "").strip()
            if not word:
                continue
            try:
                conf = float(data["conf"][i])
            except Exception:
                continue
            if conf < self.min_conf:
                continue

            block = int(data["block_num"][i])
            par = int(data["par_num"][i])
            line_num = int(data["line_num"][i])
            left = int(data["left"][i])
            top = int(data["top"][i])
            width = int(data["width"][i])
            height = int(data["height"][i])

            key = (block, par, line_num)
            grouped.setdefault(key, []).append(
                {
                    "word": word,
                    "left": left,
                    "top": top,
                    "right": left + width,
                    "bottom": top + height,
                    "height": height,
                    "conf": conf,
                    "block": block,
                }
            )

        lines = []
        for key, words in grouped.items():
            words = sorted(words, key=lambda x: x["left"])
            text = TextPreprocessor.normalize_text(" ".join(w["word"] for w in words))
            if not text:
                continue
            lines.append(
                {
                    "text": text,
                    "norm": TextPreprocessor.clean_phrase(text),
                    "left": min(w["left"] for w in words),
                    "right": max(w["right"] for w in words),
                    "top": min(w["top"] for w in words),
                    "bottom": max(w["bottom"] for w in words),
                    "height": max(w["height"] for w in words),
                    "conf": float(np.mean([w["conf"] for w in words])),
                    "block": words[0]["block"],
                }
            )

        lines.sort(key=lambda x: (x["top"], x["left"]))
        return lines

    def extract_text(self, image: np.ndarray) -> str:
        """Extract raw text from image using Tesseract.

        Args:
            image: Preprocessed binary image.

        Returns:
            Raw OCR text.
        """
        return pytesseract.image_to_string(image, lang=self.lang)
