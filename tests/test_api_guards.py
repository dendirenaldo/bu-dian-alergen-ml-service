"""Guard API batch-2: /model/train 503, _check_ready per-model,
ensemble fail-loud (RuntimeError) + ValueError input tetap 400. Tanpa model."""

from types import SimpleNamespace

import pytest


def test_model_train_dinonaktifkan_503():
    """POST /model/train tanpa body -> 503 (bukan 422/404)."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/api/v1/model/train")
    assert resp.status_code == 503
    assert "dinonaktifkan" in resp.json()["detail"]


def test_check_ready_per_model():
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from app.api.routers.detection import _check_ready

    # BERT-only: registry punya BERT, TIDAK punya BiLSTM -> harus lolos.
    bert_only = SimpleNamespace(
        is_ready=lambda m: m == "bert",
        is_loaded=False,
    )
    _check_ready(bert_only, "bert")  # tidak raise

    with pytest.raises(HTTPException) as exc:
        _check_ready(bert_only, "bilstm")
    assert exc.value.status_code == 503

    with pytest.raises(HTTPException) as exc:
        _check_ready(bert_only, "ensemble")
    assert exc.value.status_code == 503

    ready_all = SimpleNamespace(
        is_ready=lambda m: m in ("bert", "ensemble"),
        is_loaded=True,
    )
    _check_ready(ready_all, "bilstm")
    _check_ready(ready_all, "ensemble")


def test_ensemble_fail_loud_dan_nilai_input():
    pytest.importorskip("fastapi")
    from app.services.detection_pipeline import DetectionPipeline

    pipe = DetectionPipeline(registry=SimpleNamespace())

    # ValueError (input tidak valid) TIDAK boleh berubah jadi RuntimeError.
    def raise_value_error(*_a, **_k):
        raise ValueError("Teks kosong")

    pipe._classify_bilstm = raise_value_error  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="Teks kosong"):
        pipe._classify_text("x", model="ensemble")

    # Kegagalan non-input -> RuntimeError (router -> 503).
    def raise_generic(*_a, **_k):
        raise RuntimeError("model korup")

    pipe._classify_bilstm = raise_generic  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="ensemble gagal"):
        pipe._classify_text("x", model="ensemble")


def test_ensemble_vote_konsisten():
    pytest.importorskip("fastapi")
    from app.config import settings
    from app.services.detection_pipeline import DetectionPipeline

    pipe = DetectionPipeline(registry=SimpleNamespace())
    orig_strategy = settings.ENSEMBLE_STRATEGY
    try:
        settings.ENSEMBLE_STRATEGY = "vote"
        # Simulasi threshold-0.5-konsisten: unsafe iff score >= 0.5.
        pipe._classify_bilstm = lambda *_a, **_k: (  # type: ignore[method-assign]
            ("unsafe", 0.9, 0.9) if True else ("safe", 0.1, 0.1)
        )
        pipe._classify_bert = lambda *_a, **_k: ("safe", 0.9, 0.1)  # type: ignore[method-assign]
        label, conf, scores, used = pipe._classify_text("x", model="ensemble")
        assert label == "unsafe" and used == "ensemble"
        assert conf >= 0.5, "OR-vote unsafe harus punya confidence >= 0.5 (max prob >= 0.5)"
        assert scores == {"bilstm": 0.9, "bert": 0.1}
    finally:
        settings.ENSEMBLE_STRATEGY = orig_strategy
