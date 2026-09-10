import re
from typing import Dict, List, Optional

import numpy as np

from app.core.ocr.engine import OCREngine
from app.core.preprocessing.text import TextPreprocessor


class CompositionParser:
    """Parse composition/ingredients section from OCR layout."""

    ANCHOR_PATTERN = re.compile(r"\b(komposisi|composition|ingredients?)\b", re.IGNORECASE)
    STOP_PATTERN = re.compile(
        r"\b(informasi nilai gizi|nutrition|takaran saji|energi total|cara penyimpanan|penyimpanan|"
        r"netto|berat bersih|expired|kedaluwarsa|bpom|kode produksi|saran penyajian|perhatian)\b",
        re.IGNORECASE,
    )

    def __init__(self, ocr_engine: Optional[OCREngine] = None):
        self._ocr = ocr_engine or OCREngine()

    def parse(self, lines: Optional[List[Dict]] = None, image: Optional[np.ndarray] = None) -> str:
        """Parse composition text from OCR lines or image.

        If lines are not provided, runs OCR on the image first.
        Finds an anchor keyword (komposisi/composition/ingredients) and
        collects subsequent lines within the same layout region.

        Args:
            lines: Pre-computed OCR lines (optional).
            image: Preprocessed image (used if lines not provided).

        Returns:
            Extracted composition text.
        """
        if lines is None:
            if image is None:
                raise ValueError("Either lines or image must be provided")
            lines = self._ocr.extract_lines(image)

        if not lines:
            return ""

        anchors = [ln for ln in lines if self.ANCHOR_PATTERN.search(ln["text"])]
        if not anchors:
            full_text = self._ocr.extract_text(image) if image is not None else " ".join(l["text"] for l in lines)
            return TextPreprocessor.normalize_text(full_text) if full_text else ""

        anchor = anchors[0]
        avg_h = np.mean([ln["height"] for ln in lines]) if lines else 20
        max_gap = max(18, int(avg_h * 1.8))

        selected: List[str] = []
        started = False
        prev_bottom: Optional[int] = None

        for ln in lines:
            if ln["top"] < anchor["top"]:
                continue

            if not started and ln is anchor:
                started = True
                selected.append(ln["text"])
                prev_bottom = ln["bottom"]
                continue

            if not started:
                continue

            # Stop if vertical gap is too large
            if prev_bottom is not None and (ln["top"] - prev_bottom) > max_gap:
                break

            # Prefer lines in same horizontal region as anchor
            horizontal_overlap = not (ln["right"] < anchor["left"] - 100 or ln["left"] > anchor["right"] + 900)
            near_anchor_column = abs(ln["left"] - anchor["left"]) <= 220
            if not (horizontal_overlap or near_anchor_column):
                continue

            if self.STOP_PATTERN.search(ln["text"]):
                break

            selected.append(ln["text"])
            prev_bottom = ln["bottom"]

        comp = TextPreprocessor.normalize_text(" ".join(selected))
        comp = re.sub(
            r"(?i)^\s*(komposisi|composition|ingredients?)\s*[:\-]?\s*", "", comp
        ).strip()
        return comp
