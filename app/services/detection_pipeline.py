import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import tensorflow as tf

from app.config import settings
from app.core.preprocessing.image import ImagePreprocessor
from app.core.preprocessing.text import TextPreprocessor
from app.core.ocr.composition import CompositionParser
from app.core.ocr.engine import OCREngine
from app.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

KNOWN_ALLERGENS = {
    "gluten": ["wheat", "barley", "rye", "oats", "gluten", "semolina", "spelt"],
    "dairy": ["milk", "cheese", "butter", "cream", "yogurt", "lactose", "casein", "whey"],
    "egg": ["egg", "albumin", "lysozyme"],
    "soy": ["soy", "soya", "lecithin"],
    "peanut": ["peanut"],
    "tree_nut": ["almond", "cashew", "walnut", "hazelnut", "pecan", "pistachio", "macadamia"],
    "fish": ["fish", "anchovy", "cod", "bass", "salmon", "tilapia", "tuna"],
    "shellfish": ["shrimp", "crab", "lobster", "mussel", "clam", "oyster", "squid"],
    "sesame": ["sesame", "tahini"],
    "celery": ["celery"],
    "mustard": ["mustard"],
    "sulfite": ["sulfite", "sulphite", "sodium sulfite"],
}


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
        start = time.time()

        img_prep = ImagePreprocessor()
        processed_img = img_prep.preprocess_from_bytes(image_bytes)

        ocr = OCREngine(lang=settings.TESSERACT_LANG, min_conf=settings.OCR_MIN_CONF)
        lines = ocr.extract_lines(processed_img)

        parser = CompositionParser()
        composition = parser.parse(lines, image=processed_img)

        if not composition:
            composition = ocr.extract_text(processed_img)

        result, confidence = self._classify_text(composition)
        allergens = self._identify_allergens(composition) if result == "unsafe" else []

        elapsed = int((time.time() - start) * 1000)

        return DetectionResult(
            result=result,
            confidence_score=confidence,
            ocr_text=composition,
            allergens=allergens,
            processing_time_ms=elapsed,
            detection_method="image_ocr",
        )

    def detect_from_text(self, text: str) -> DetectionResult:
        start = time.time()

        result, confidence = self._classify_text(text)
        allergens = self._identify_allergens(text) if result == "unsafe" else []

        elapsed = int((time.time() - start) * 1000)

        return DetectionResult(
            result=result,
            confidence_score=confidence,
            ocr_text=text,
            allergens=allergens,
            processing_time_ms=elapsed,
            detection_method="text_input",
        )

    def _classify_text(self, text: str) -> "tuple[str, float]":
        model, tokenizer, label_encoder, loaded = self.registry.snapshot()
        if not loaded or model is None or tokenizer is None:
            raise RuntimeError("Models not loaded")

        preprocessor = TextPreprocessor()
        tokens = preprocessor.preprocess(text)
        cleaned = " ".join(tokens)
        if not cleaned.strip():
            raise ValueError("Teks kosong setelah preprocessing")

        pad_len = settings.MAX_LEN

        seq = tokenizer.texts_to_sequences([cleaned])
        pad = tf.keras.preprocessing.sequence.pad_sequences(
            seq, maxlen=pad_len, padding="post", truncating="post"
        )

        pred = self.registry.bi_lstm_model.predict(pad, verbose=0)
        score = float(pred[0][0])

        # Gunakan label_encoder bila tersedia agar mapping tidak brittle.
        # Fallback alfabetis lama: index 1 = unsafe.
        unsafe_idx = 0
        try:
            if label_encoder is not None and hasattr(label_encoder, "classes_"):
                classes = list(label_encoder.classes_)
                unsafe_idx = classes.index("unsafe") if "unsafe" in classes else 0
        except Exception:
            unsafe_idx = 0

        if pred.shape[-1] > 1:
            score = float(pred[0][unsafe_idx])
            label = "unsafe" if int(pred.argmax(axis=-1)[0]) == unsafe_idx else "safe"
            confidence = score if label == "unsafe" else 1 - score
        else:
            label = "unsafe" if score >= 0.5 else "safe"
            confidence = score if score >= 0.5 else 1 - score

        return label, confidence

    def _identify_allergens(self, text: str) -> List[AllergenResult]:
        text_lower = text.lower()
        found = []
        for allergen_name, keywords in KNOWN_ALLERGENS.items():
            for kw in keywords:
                if kw in text_lower:
                    found.append(AllergenResult(
                        name=allergen_name,
                        confidence=1.0,
                        severity="high",
                    ))
                    break
        return found
