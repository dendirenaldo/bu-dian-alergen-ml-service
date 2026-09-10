import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import tensorflow as tf

from app.core.preprocessing.image import ImagePreprocessor
from app.core.preprocessing.text import TextPreprocessor
from app.core.ocr.composition import CompositionParser
from app.core.ocr.engine import OCREngine
from app.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class AllergenResult:
    name: str
    confidence: float
    severity: str


@dataclass
class DetectionResult:
    result: str
    confidence_score: float
    ocr_text: Optional[str]
    allergens: List[AllergenResult]
    processing_time_ms: int
    detection_method: str


class DetectionPipeline:
    """End-to-end allergen detection pipeline."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def detect_from_image(self, image_bytes: bytes) -> DetectionResult:
        """Run full detection pipeline on an image.

        Preprocesses the image, extracts text via OCR, parses composition,
        and classifies using the BiLSTM model.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            DetectionResult with classification and metadata.

        Raises:
            RuntimeError: If models are not loaded.
            ValueError: If image cannot be decoded.
        """
        start = time.time()

        img_prep = ImagePreprocessor()
        processed_img = img_prep.preprocess_from_bytes(image_bytes)

        ocr = OCREngine()
        lines = ocr.extract_lines(processed_img)

        parser = CompositionParser()
        composition = parser.parse(lines, image=processed_img)

        if not composition:
            composition = ocr.extract_text(processed_img)

        result, confidence = self._classify_text(composition)

        elapsed = int((time.time() - start) * 1000)

        return DetectionResult(
            result=result,
            confidence_score=confidence,
            ocr_text=composition,
            allergens=[],
            processing_time_ms=elapsed,
            detection_method="image_ocr",
        )

    def detect_from_text(self, text: str) -> DetectionResult:
        """Classify raw text for allergen presence.

        Args:
            text: Ingredient/composition text.

        Returns:
            DetectionResult with classification and metadata.

        Raises:
            RuntimeError: If models are not loaded.
        """
        start = time.time()

        result, confidence = self._classify_text(text)

        elapsed = int((time.time() - start) * 1000)

        return DetectionResult(
            result=result,
            confidence_score=confidence,
            ocr_text=text,
            allergens=[],
            processing_time_ms=elapsed,
            detection_method="text_input",
        )

    def _classify_text(self, text: str) -> "tuple[str, float]":
        """Run BiLSTM classification on text.

        Args:
            text: Preprocessed text to classify.

        Returns:
            Tuple of (label, confidence).

        Raises:
            RuntimeError: If models are not loaded.
        """
        if not self.registry.is_loaded:
            raise RuntimeError("Models not loaded")

        preprocessor = TextPreprocessor()
        tokens = preprocessor.preprocess(text)
        cleaned = " ".join(tokens)

        tokenizer = self.registry.tokenizer
        pad_len = 120

        seq = tokenizer.texts_to_sequences([cleaned])
        pad = tf.keras.preprocessing.sequence.pad_sequences(
            seq, maxlen=pad_len, padding="post"
        )

        pred = self.registry.bi_lstm_model.predict(pad, verbose=0)
        score = float(pred[0][0])

        label = "unsafe" if score >= 0.5 else "safe"
        confidence = score if score >= 0.5 else 1 - score

        return label, confidence
