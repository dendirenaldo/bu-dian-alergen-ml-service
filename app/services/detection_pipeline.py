import logging
import re
import time
from dataclasses import dataclass
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

# FIX (HIGH): kamus ID + EN dengan word-boundary (hindari 'egg in eggplant').
KNOWN_ALLERGENS = {
    "gluten": ["wheat", "barley", "rye", "oats", "gluten", "semolina", "spelt",
               "gandum", "terigu"],
    "dairy": ["milk", "cheese", "butter", "cream", "yogurt", "lactose", "casein",
              "whey", "susu", "keju", "mentega", "krim", "laktosa"],
    "egg": ["egg", "albumin", "lysozyme", "telur"],
    "soy": ["soy", "soya", "lecithin", "kedelai", "kedele", "lesitin"],
    "peanut": ["peanut", "kacang tanah"],
    "tree_nut": ["almond", "cashew", "walnut", "hazelnut", "pecan", "pistachio",
                 "macadamia", "kacang mete", "kacang almond", "kemiri", "kenari"],
    "fish": ["fish", "anchovy", "cod", "bass", "salmon", "tilapia", "tuna",
             "ikan", "teri", "tongkol", "tuna"],
    "shellfish": ["shrimp", "crab", "lobster", "mussel", "clam", "oyster",
                  "squid", "udang", "kepiting", "kerang", "cumi"],
    "sesame": ["sesame", "tahini", "wijen"],
    "celery": ["celery", "seledri"],
    "mustard": ["mustard", "mostar", "mustar"],
    "sulfite": ["sulfite", "sulphite", "sodium sulfite", "sulfit"],
}

# Precompile word-boundary regex per keyword.
_ALLERGEN_PATTERNS: list[tuple[str, "re.Pattern"]] = []
for _name, _kws in KNOWN_ALLERGENS.items():
    for _kw in _kws:
        _ALLERGEN_PATTERNS.append(
            (_name, re.compile(r"\b" + re.escape(_kw.lower()) + r"\b"))
        )


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
    model_name: str = "bilstm"
    model_version: str = ""
    scores: Optional[dict] = None


class DetectionPipeline:
    """End-to-end allergen detection pipeline (dual-model BiLSTM + BERT)."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        # FIX (HIGH): singleton preprocessor — jangan buat per request.
        self._preprocessor = TextPreprocessor()

    def detect_from_image(
        self, image_bytes: bytes, model: str = "bilstm"
    ) -> DetectionResult:
        start = time.time()

        img_prep = ImagePreprocessor()
        processed_img = img_prep.preprocess_from_bytes(image_bytes)

        ocr = OCREngine(lang=settings.TESSERACT_LANG, min_conf=settings.OCR_MIN_CONF)
        lines = ocr.extract_lines(processed_img)

        # FIX: teruskan engine berkonfigurasi ke parser agar fallback
        # extract_text memakai lang/min_conf yang sama.
        parser = CompositionParser(ocr_engine=ocr)
        composition = parser.parse(lines, image=processed_img)

        if not composition:
            composition = ocr.extract_text(processed_img)

        result, confidence, scores, used_model = self._classify_text(composition, model)
        # FIX: alergen dihitung independen dari label model (tidak disembunyikan
        # saat model memprediksi safe) — skor model tetap jadi penentu label.
        allergens = self._identify_allergens(composition, confidence)

        elapsed = int((time.time() - start) * 1000)

        return DetectionResult(
            result=result,
            confidence_score=confidence,
            ocr_text=composition,
            allergens=allergens,
            processing_time_ms=elapsed,
            detection_method="image_ocr" if used_model == "bilstm" else f"image_ocr_{used_model}",
            model_name=used_model,
            scores=scores,
        )

    def detect_from_text(
        self, text: str, model: str = "bilstm"
    ) -> DetectionResult:
        start = time.time()

        result, confidence, scores, used_model = self._classify_text(text, model)
        allergens = self._identify_allergens(text, confidence)

        elapsed = int((time.time() - start) * 1000)

        return DetectionResult(
            result=result,
            confidence_score=confidence,
            ocr_text=text,
            allergens=allergens,
            processing_time_ms=elapsed,
            detection_method="text_input" if used_model == "bilstm" else f"text_input_{used_model}",
            model_name=used_model,
            scores=scores,
        )

    # -- klasifikasi dual-model --
    def _classify_text(self, text: str, model: str = "bilstm") -> "tuple[str, float, dict, str]":
        model = (model or "bilstm").lower()
        if model not in ("bilstm", "bert", "ensemble"):
            raise ValueError("model harus salah satu: bilstm, bert, ensemble")
        if model == "bilstm":
            label, conf, score = self._classify_bilstm(text)
            return label, conf, {"bilstm": score}, "bilstm"
        if model == "bert":
            label, conf, score = self._classify_bert(text)
            return label, conf, {"bert": score}, "bert"
        # ensemble
        try:
            b_label, b_conf, b_score = self._classify_bilstm(text)
        except Exception as e:
            logger.warning("BiLSTM ensemble gagal, fallback BERT: %s", e)
            label, conf, score = self._classify_bert(text)
            return label, conf, {"bert": score}, "bert"
        try:
            t_label, t_conf, t_score = self._classify_bert(text)
        except Exception as e:
            logger.warning("BERT ensemble gagal, fallback BiLSTM: %s", e)
            return b_label, b_conf, {"bilstm": b_score}, "bilstm"
        if (settings.ENSEMBLE_STRATEGY or "weighted") == "vote":
            votes = [1 if b_label == "unsafe" else 0, 1 if t_label == "unsafe" else 0]
            unsafe = sum(votes) >= 1  # OR-vote: utamakan recall alergen
            score = (b_score + t_score) / 2.0
        else:
            wb = float(settings.ENSEMBLE_WEIGHT_BILSTM)
            wt = float(settings.ENSEMBLE_WEIGHT_BERT)
            s = (wb + wt) or 1.0
            score = (wb * b_score + wt * t_score) / s
            unsafe = score >= 0.5
        label = "unsafe" if unsafe else "safe"
        conf = score if unsafe else 1 - score
        return label, float(conf), {"bilstm": float(b_score), "bert": float(t_score)}, "ensemble"

    def _classify_bilstm(self, text: str) -> "tuple[str, float, float]":
        model, tokenizer, label_encoder, loaded = self.registry.snapshot()
        if not loaded or model is None or tokenizer is None:
            raise RuntimeError("Models not loaded")

        tokens = self._preprocessor.preprocess(text)
        cleaned = " ".join(tokens)
        if not cleaned.strip():
            raise ValueError("Teks kosong setelah preprocessing")

        pad_len = settings.MAX_LEN
        # OOV guard: cap indeks >= vocab agar tidak IndexError.
        vocab = getattr(settings, "VOCAB_SIZE", 20000)
        seq = tokenizer.texts_to_sequences([cleaned])
        seq = [[i if i < vocab else 1 for i in s] for s in seq]
        pad = tf.keras.preprocessing.sequence.pad_sequences(
            seq, maxlen=pad_len, padding="post", truncating="post"
        )

        # FIX TOCTOU: pakai snapshot `model`, bukan properti registry.
        pred = model.predict(pad, verbose=0)
        raw = float(pred[0][0]) if pred.shape[-1] == 1 else None

        unsafe_idx = 0
        try:
            if label_encoder is not None and hasattr(label_encoder, "classes_"):
                classes = list(label_encoder.classes_)
                unsafe_idx = classes.index("unsafe") if "unsafe" in classes else 0
        except Exception:
            unsafe_idx = 0

        if pred.shape[-1] > 1:
            score = float(pred[0][unsafe_idx])
        else:
            score = float(raw if raw is not None else 0.0)
        threshold = self.registry.get_threshold("bilstm", settings.DEFAULT_THRESHOLD)
        label = "unsafe" if score >= threshold else "safe"
        confidence = score if label == "unsafe" else 1 - score
        return label, float(confidence), float(score)

    def _classify_bert(self, text: str) -> "tuple[str, float, float]":
        bert_model, bert_tok, bert_thr, _name = self.registry.bert_snapshot()
        if bert_model is None or bert_tok is None:
            raise RuntimeError("Model BERT belum dimuat (cek BERT_MODEL_DIR)")
        cleaned = (text or "").strip()
        if not cleaned:
            raise ValueError("Teks kosong")
        try:
            import torch
        except ImportError as e:
            raise RuntimeError("torch belum terinstal untuk inferensi BERT") from e
        enc = bert_tok(
            cleaned, truncation=True, padding=True,
            max_length=getattr(settings, "BERT_MAX_LEN", 256),
            return_tensors="pt",
        )
        bert_model.eval()
        with torch.no_grad():
            logits = bert_model(**enc).logits.detach().cpu().numpy().ravel()
        import math
        score = float(1 / (1 + math.exp(-float(logits[0]))))
        threshold = float(bert_thr or getattr(settings, "BERT_THRESHOLD", 0.5))
        label = "unsafe" if score >= threshold else "safe"
        confidence = score if label == "unsafe" else 1 - score
        return label, float(confidence), float(score)

    def _identify_allergens(self, text: str, model_confidence: float = 1.0) -> List[AllergenResult]:
        text_lower = (text or "").lower()
        found = []
        for allergen_name, pattern in _ALLERGEN_PATTERNS:
            if pattern.search(text_lower):
                if not any(r.name == allergen_name for r in found):
                    found.append(AllergenResult(
                        name=allergen_name,
                        # Propagasi keyakinan model alih-alih 1.0 hardcoded.
                        confidence=round(float(model_confidence), 3),
                        severity="high",
                    ))
        return found
