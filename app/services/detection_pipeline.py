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
    model_name: str = "bert"
    model_version: str = ""
    scores: Optional[dict] = None
    tahap_ms: Optional[dict] = None


def result_to_dict(r: "DetectionResult") -> dict:
    """Serialisasi DetectionResult untuk event SSE done / response."""
    return {
        "result": r.result,
        "confidence_score": r.confidence_score,
        "ocr_text": r.ocr_text,
        "allergens": [
            {"name": a.name, "confidence": a.confidence, "severity": a.severity}
            for a in (r.allergens or [])
        ],
        "processing_time_ms": r.processing_time_ms,
        "detection_method": r.detection_method,
        "model_name": r.model_name,
        "model_version": r.model_version,
        "scores": r.scores,
        "tahap_ms": r.tahap_ms or {},
    }


def _drain_stream(gen):
    """Konsumsi generator tahap sampai return DetectionResult (PEP 380)."""
    result = None
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            result = stop.value
            break
    return result


class DetectionPipeline:
    """End-to-end allergen detection pipeline (dual-model BiLSTM + BERT)."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._preprocessor = TextPreprocessor()

    def _stage(self, stage: str, progress: int, message: str,
                 data: Optional[dict] = None, tahap_ms: int = 0,
                 eta_ms: Optional[int] = None) -> dict:
        """Bentuk event tahap SSE {stage, progress, message, etaMs?, data?}."""
        event: dict = {"stage": stage, "progress": progress, "message": message}
        if eta_ms is not None:
            event["etaMs"] = eta_ms
        payload = dict(data or {})
        payload["tahap_ms"] = int(tahap_ms)
        event["data"] = payload
        return event

    def detect_from_image_stream(self, image_bytes: bytes, model: str = "bert"):
        """Generator tahap image: preprocess 30 → ocr 55 → classify 80 → mapping 90.

        Me-return DetectionResult (StopIteration.value) setelah yield terakhir.
        Tiap event membawa tahap_ms tahap tsb + elapsed_ms kumulatif.
        """
        t0 = time.time()

        s = time.time()
        img_prep = ImagePreprocessor()
        processed_img = img_prep.preprocess_from_bytes(image_bytes)
        ms_pre = int((time.time() - s) * 1000)
        yield self._stage("preprocess", 30, "Preprocessing gambar selesai",
                          {"elapsed_ms": int((time.time() - t0) * 1000)}, ms_pre, 9000)

        s = time.time()
        ocr = OCREngine(lang=settings.TESSERACT_LANG, min_conf=settings.OCR_MIN_CONF)
        lines = ocr.extract_lines(processed_img)

        parser = CompositionParser(ocr_engine=ocr)
        composition = parser.parse(lines, image=processed_img)

        if not composition:
            composition = ocr.extract_text(processed_img)
        ms_ocr = int((time.time() - s) * 1000)
        line_count = len(lines or [])
        if line_count == 0 and composition:
            line_count = len([ln for ln in composition.splitlines() if ln.strip()])
        yield self._stage("ocr", 55, f"OCR selesai ({line_count} baris)",
                          {"lines": line_count,
                           "elapsed_ms": int((time.time() - t0) * 1000)}, ms_ocr, 6000)

        s = time.time()
        result, confidence, scores, used_model = self._classify_text(composition, model)
        ms_cls = int((time.time() - s) * 1000)
        yield self._stage("classify", 80, f"Klasifikasi selesai (backend {used_model})",
                          {"backend": used_model, "model": used_model,
                           "elapsed_ms": int((time.time() - t0) * 1000)}, ms_cls, 3000)

        s = time.time()
        allergens = self._identify_allergens(composition, confidence)
        ms_map = int((time.time() - s) * 1000)
        yield self._stage("mapping", 90, f"Pemetaan alergen selesai ({len(allergens)})",
                          {"count": len(allergens),
                           "elapsed_ms": int((time.time() - t0) * 1000)}, ms_map, 1000)

        elapsed = int((time.time() - t0) * 1000)
        tahap = {"preprocess": ms_pre, "ocr": ms_ocr,
                 "classify": ms_cls, "mapping": ms_map}

        return DetectionResult(
            result=result,
            confidence_score=confidence,
            ocr_text=composition,
            allergens=allergens,
            processing_time_ms=elapsed,
            detection_method="image_ocr" if used_model == "bilstm" else f"image_ocr_{used_model}",
            model_name=used_model,
            scores=scores,
            tahap_ms=tahap,
        )

    def detect_from_text_stream(self, text: str, model: str = "bert"):
        """Generator tahap text: preprocess 30 → classify 80 → mapping 90."""
        t0 = time.time()

        s = time.time()
        cleaned = (text or "").strip()
        ms_pre = int((time.time() - s) * 1000)
        yield self._stage("preprocess", 30, "Preprocessing teks selesai",
                          {"elapsed_ms": int((time.time() - t0) * 1000)}, ms_pre, 5000)

        s = time.time()
        result, confidence, scores, used_model = self._classify_text(text, model)
        ms_cls = int((time.time() - s) * 1000)
        yield self._stage("classify", 80, f"Klasifikasi selesai (backend {used_model})",
                          {"backend": used_model, "model": used_model,
                           "elapsed_ms": int((time.time() - t0) * 1000)}, ms_cls, 3000)

        s = time.time()
        allergens = self._identify_allergens(cleaned, confidence)
        ms_map = int((time.time() - s) * 1000)
        yield self._stage("mapping", 90, f"Pemetaan alergen selesai ({len(allergens)})",
                          {"count": len(allergens),
                           "elapsed_ms": int((time.time() - t0) * 1000)}, ms_map, 1000)

        elapsed = int((time.time() - t0) * 1000)
        tahap = {"preprocess": ms_pre, "ocr": 0,
                 "classify": ms_cls, "mapping": ms_map}

        return DetectionResult(
            result=result,
            confidence_score=confidence,
            ocr_text=text,
            allergens=allergens,
            processing_time_ms=elapsed,
            detection_method="text_input" if used_model == "bilstm" else f"text_input_{used_model}",
            model_name=used_model,
            scores=scores,
            tahap_ms=tahap,
        )

    def detect_from_image(
        self, image_bytes: bytes, model: str = "bert"
    ) -> DetectionResult:
        """Wrapper sync untuk POST lama (menguras generator tahap)."""
        return _drain_stream(self.detect_from_image_stream(image_bytes, model=model))

    def detect_from_text(
        self, text: str, model: str = "bert"
    ) -> DetectionResult:
        """Wrapper sync untuk POST lama (menguras generator tahap)."""
        return _drain_stream(self.detect_from_text_stream(text, model=model))

    def _classify_text(self, text: str, model: str = "bert") -> "tuple[str, float, dict, str]":
        model = (model or "bert").lower()
        if model not in ("bilstm", "bert", "ensemble"):
            raise ValueError("model harus salah satu: bilstm, bert, ensemble")
        if model == "bilstm":
            label, conf, score = self._classify_bilstm(text)
            return label, conf, {"bilstm": score}, "bilstm"
        if model == "bert":
            label, conf, score = self._classify_bert(text)
            return label, conf, {"bert": score}, "bert"
        try:
            b_label, b_conf, b_score = self._classify_bilstm(text)
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Cabang BiLSTM ensemble gagal: {e}") from e
        try:
            t_label, t_conf, t_score = self._classify_bert(text)
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Cabang BERT ensemble gagal: {e}") from e
        if (settings.ENSEMBLE_STRATEGY or "weighted") == "vote":
            unsafe = b_label == "unsafe" or t_label == "unsafe"
            score = max(b_score, t_score)
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

        raw = (text or "").strip()
        if not raw:
            raise ValueError("Teks kosong")

        pad_len = settings.MAX_LEN
        vocab = getattr(settings, "VOCAB_SIZE", 20000)
        seq = tokenizer.texts_to_sequences([text])
        if not seq or not seq[0]:
            raise ValueError("Teks kosong setelah tokenisasi")
        seq = [[i if i < vocab else 1 for i in s] for s in seq]
        pad = tf.keras.preprocessing.sequence.pad_sequences(
            seq, maxlen=pad_len, padding="post", truncating="post"
        )

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
        if logits.size == 1:
            import math
            score = float(1 / (1 + math.exp(-float(logits[0]))))
        else:
            id2label = getattr(getattr(bert_model, "config", None), "id2label", None) or {}
            labels = [str(v).lower() for v in id2label.values()]
            try:
                unsafe_idx = labels.index("unsafe")
            except ValueError:
                unsafe_idx = 1
            shifted = logits - float(np.max(logits))
            exp = np.exp(shifted)
            score = float(exp[unsafe_idx] / exp.sum())
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
                        confidence=round(float(model_confidence), 3),
                        severity="high",
                    ))
        return found
