"""Regresi parity serving-vs-training untuk BiLSTM V5.

- Tanpa TF: pastikan preprocess_v5 TIDAK membuang stopwords
  (skew yang pernah terjadi di _classify_bilstm).
- Butuh TF (+models): adapter JSON tokenizer + parity skor penuh.
"""

import os

import pytest


def test_preprocess_v5_tanpa_stopword():
    from app.core.preprocessing.text import TextPreprocessor

    pp = TextPreprocessor()
    toks = pp.preprocess_v5("Susu dan Telur, Mengandung 8%!")
    assert "dan" in toks  # stopword dipertahankan (parity training)
    assert "susu" in toks and "telur" in toks
    assert all(t == t.lower() for t in toks)


def test_json_tokenizer_oov_dicap():
    tf = pytest.importorskip("tensorflow")
    from app.services.model_registry import _load_tokenizer_json
    import json

    cfg = {"vocab_size": 10, "max_len": 8,
           "word_index": {"susu": 2, "telur": 3}}
    p = "/tmp/_tok_test.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    tok = _load_tokenizer_json(p)
    seqs = tok.texts_to_sequences(["susu telur kataasing"])
    assert all(i < 10 for s in seqs for i in s)
    assert seqs[0][:2] == [2, 3]


def test_registry_parity_vs_training():
    pytest.importorskip("tensorflow")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    tri_csv = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "bu-dian-alergen-ml-training", "artifacts", "bilstm", "output",
        "gold_kb_bilstm_triage.csv",
    )
    if not os.path.exists(os.path.join(models_dir, "bilstm_model.keras")):
        pytest.skip("artefak serving belum di-export")
    if not os.path.exists(tri_csv):
        pytest.skip("triage training tidak ada")
    import pandas as pd

    from app.services.detection_pipeline import DetectionPipeline
    from app.services.model_registry import ModelRegistry

    reg = ModelRegistry()
    reg.reset()
    reg.load_models(models_dir)
    assert reg.is_ready("bilstm")
    assert reg.get_threshold("bilstm", 0.5) == 0.5
    pipe = DetectionPipeline(reg)
    tri = pd.read_csv(tri_csv).sample(3, random_state=7)
    for _, r in tri.iterrows():
        res = pipe.detect_from_text(r["text"], model="bilstm")
        assert abs(res.scores["bilstm"] - r["prob_unsafe"]) < 1e-6
        assert res.result == ("unsafe" if r["prob_unsafe"] >= 0.5 else "safe")
    reg.reset()
