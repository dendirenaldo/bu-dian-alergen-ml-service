import re
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytesseract

from app.core.preprocessing.text import TextPreprocessor


class BoldDetector:
    """Detect bold phrases in images and match them against composition items."""

    def __init__(self, min_conf: int = 45):
        self.min_conf = min_conf

    def extract_bold_phrases(self, image_path: str) -> List[str]:
        """Extract bold phrases from an image using ink ratio analysis.

        Uses adaptive thresholding and per-word ink density to identify
        bold text, then groups bold words into phrases.

        Args:
            image_path: Path to the image file.

        Returns:
            List of unique bold phrases.
        """
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bin_inv = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )

        data = pytesseract.image_to_data(gray, lang="eng", output_type=pytesseract.Output.DICT)

        words: List[Dict] = []
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

            l = int(data["left"][i])
            t = int(data["top"][i])
            w = max(1, int(data["width"][i]))
            h = max(1, int(data["height"][i]))
            roi = bin_inv[t : t + h, l : l + w]
            ink_ratio = float(np.mean(roi > 0)) if roi.size else 0.0

            words.append(
                {
                    "word": word,
                    "ink": ink_ratio,
                    "h": h,
                    "left": l,
                    "block": int(data["block_num"][i]),
                    "par": int(data["par_num"][i]),
                    "line": int(data["line_num"][i]),
                }
            )

        if not words:
            return []

        ink_thr = float(np.percentile([w["ink"] for w in words], 75))
        h_thr = float(np.percentile([w["h"] for w in words], 60))
        bold_words = [w for w in words if w["ink"] >= ink_thr and w["h"] >= h_thr]

        grouped: Dict[tuple, List[Dict]] = {}
        for w in bold_words:
            grouped.setdefault((w["block"], w["par"], w["line"]), []).append(w)

        phrases: List[str] = []
        for _, ws in grouped.items():
            ws = sorted(ws, key=lambda x: x["left"])
            phrase = TextPreprocessor.normalize_text(" ".join(x["word"] for x in ws))
            if len(TextPreprocessor.clean_phrase(phrase)) >= 2:
                phrases.append(phrase)

        seen: set = set()
        out: List[str] = []
        for p in phrases:
            k = TextPreprocessor.clean_phrase(p)
            if k and k not in seen:
                seen.add(k)
                out.append(p)
        return out

    @staticmethod
    def split_composition_items(composition_text: str) -> List[str]:
        """Split composition text into individual items.

        Args:
            composition_text: Raw composition string.

        Returns:
            List of individual composition items.
        """
        raw_items = re.split(r"[,;•|]", composition_text)
        items: List[str] = []
        for item in raw_items:
            s = TextPreprocessor.normalize_text(item)
            if len(s) >= 2:
                items.append(s)
        return items

    @staticmethod
    def token_overlap_score(a: str, b: str) -> float:
        """Compute token overlap ratio between two strings.

        Args:
            a: First string.
            b: Second string.

        Returns:
            Overlap ratio between 0 and 1.
        """
        sa = set(TextPreprocessor.clean_phrase(a).split())
        sb = set(TextPreprocessor.clean_phrase(b).split())
        if not sa or not sb:
            return 0.0
        inter = len(sa.intersection(sb))
        return inter / max(1, min(len(sa), len(sb)))

    def detect_bold_composition_items(
        self, image_path: str, composition_text: str
    ) -> Tuple[str, List[str]]:
        """Detect which composition items appear in bold text.

        Matches bold phrases from OCR against composition items using
        token overlap and substring containment.

        Args:
            image_path: Path to the image file.
            composition_text: Parsed composition text.

        Returns:
            Tuple of (label, list of bold items).
            Label is 'unsafe' if bold items found, 'safe' otherwise.
        """
        bold_phrases = self.extract_bold_phrases(image_path)
        composition_items = self.split_composition_items(composition_text)

        bold_items: List[str] = []
        for item in composition_items:
            item_clean = TextPreprocessor.clean_phrase(item)
            if not item_clean:
                continue
            for bp in bold_phrases:
                bp_clean = TextPreprocessor.clean_phrase(bp)
                if not bp_clean:
                    continue
                overlap = self.token_overlap_score(item, bp)
                contains = item_clean in bp_clean or bp_clean in item_clean
                if overlap >= 0.6 or contains:
                    bold_items.append(item)
                    break

        uniq: List[str] = []
        seen: set = set()
        for x in bold_items:
            k = TextPreprocessor.clean_phrase(x)
            if k and k not in seen:
                seen.add(k)
                uniq.append(x)

        label = "unsafe" if uniq else "safe"
        return label, uniq
