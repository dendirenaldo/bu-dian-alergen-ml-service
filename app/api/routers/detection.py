import asyncio
import json
import logging
import os
import time
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import StreamingResponse

from app.api.deps import get_api_key, get_model_registry
from app.schemas.detection import (
    AllergenResultSchema,
    DetectionResponse,
    TextDetectionRequest,
)
from app.services.detection_pipeline import DetectionPipeline, result_to_dict
from app.services.model_registry import ModelRegistry


def _to_allergen_schemas(allergens) -> list:
    """Konversi dataclass pipeline -> schema API (pydantic v2 ketat)."""
    out = []
    for a in allergens or []:
        if isinstance(a, dict):
            out.append(AllergenResultSchema(**a))
        else:
            out.append(AllergenResultSchema(
                name=getattr(a, "name", ""),
                confidence=float(getattr(a, "confidence", 0.0)),
                severity=getattr(a, "severity", ""),
            ))
    return out


logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB

# Direktori penyimpanan bytes upload agar GET /detection/stream?jobId bisa
# membaca ulang file dari disk bila jobId diketahui.
UPLOAD_DIRS = ("./uploads", "/tmp/ml_uploads", "/tmp")
HEARTBEAT_INTERVAL_S = 15.0


def _save_job_bytes(data: bytes) -> Optional[str]:
    """Simpan bytes upload ke disk, kembalikan jobId (None bila gagal)."""
    try:
        os.makedirs(UPLOAD_DIRS[0], exist_ok=True)
        job_id = uuid.uuid4().hex
        with open(os.path.join(UPLOAD_DIRS[0], f"{job_id}.bin"), "wb") as f:
            f.write(data)
        return job_id
    except OSError:
        return None


def _find_job_bytes(job_id: str) -> Optional[bytes]:
    """Baca bytes dari disk berdasarkan jobId; None bila tidak ada."""
    safe = "".join(c for c in (job_id or "") if c.isalnum() or c in ("-", "_"))[:128]
    if not safe:
        return None
    for d in UPLOAD_DIRS:
        for cand in (os.path.join(d, safe), os.path.join(d, f"{safe}.bin")):
            if os.path.isfile(cand):
                try:
                    with open(cand, "rb") as f:
                        return f.read()
                except OSError:
                    return None
        try:
            for name in os.listdir(d):
                if name.startswith(safe):
                    p = os.path.join(d, name)
                    if os.path.isfile(p):
                        with open(p, "rb") as f:
                            return f.read()
        except OSError:
            continue
    return None


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _error_event(message: str, progress: int = 0, data: Optional[dict] = None) -> dict:
    return {"stage": "error", "progress": progress, "message": message,
            "data": data or {}}

router = APIRouter()

ModelChoice = Literal["bilstm", "bert", "ensemble"]


def _check_ready(registry: ModelRegistry, model: str) -> None:
    if model == "bert":
        if not registry.is_ready("bert"):
            raise HTTPException(status_code=503, detail="Model BERT belum dimuat")
        return
    if model == "ensemble" and not registry.is_ready("ensemble"):
        raise HTTPException(
            status_code=503,
            detail="Ensemble butuh BiLSTM+BERT; salah satu belum dimuat",
        )
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")


@router.post("/detection/upload", response_model=DetectionResponse)
async def upload_image(
    file: UploadFile | None = File(default=None),
    image: UploadFile | None = File(default=None),
    model: ModelChoice = Query(default="bert", description="Model: bilstm|bert|ensemble"),
    registry: ModelRegistry = Depends(get_model_registry),
    _api_key: str | None = Depends(get_api_key),
    response: Response | None = None,
):
    _check_ready(registry, model)

    upload = file or image
    if upload is None:
        raise HTTPException(status_code=422, detail="Field 'file' (atau 'image') wajib diisi")

    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await upload.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large, max 10MB")

    try:
        import cv2
        import numpy as np

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is None:
            raise HTTPException(status_code=400, detail="File gambar rusak/tidak valid")
    except HTTPException:
        raise
    except Exception:
        pass

    try:
        pipeline = DetectionPipeline(registry)
        result = pipeline.detect_from_image(image_bytes, model=model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e) or "Input tidak valid")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e) or "Models not loaded")
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail="Detection failed")

    if response is not None:
        job_id = _save_job_bytes(image_bytes)
        if job_id:
            response.headers["X-Job-Id"] = job_id

    return DetectionResponse(
        result=result.result,
        confidence_score=result.confidence_score,
        ocr_text=result.ocr_text,
        allergens=_to_allergen_schemas(result.allergens),
        processing_time_ms=result.processing_time_ms,
        detection_method=result.detection_method,
        model_name=result.model_name,
        model_version=result.model_version,
        scores=result.scores,
        tahap_ms=result.tahap_ms,
    )


@router.post("/detection/text", response_model=DetectionResponse)
async def classify_text(
    request: TextDetectionRequest,
    model: ModelChoice = Query(default="bert", description="Model: bilstm|bert|ensemble"),
    registry: ModelRegistry = Depends(get_model_registry),
    _api_key: str = Depends(get_api_key),
):
    _check_ready(registry, model)

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        pipeline = DetectionPipeline(registry)
        result = pipeline.detect_from_text(request.text, model=model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e) or "Input tidak valid")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e) or "Models not loaded")
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail="Classification failed")

    return DetectionResponse(
        result=result.result,
        confidence_score=result.confidence_score,
        ocr_text=result.ocr_text,
        allergens=_to_allergen_schemas(result.allergens),
        processing_time_ms=result.processing_time_ms,
        detection_method=result.detection_method,
        model_name=result.model_name,
        model_version=result.model_version,
        scores=result.scores,
        tahap_ms=result.tahap_ms,
    )


@router.get("/detection/stream")
async def stream_detection(
    jobId: str | None = Query(default=None),
    job_id: str | None = Query(default=None),
    model: ModelChoice = Query(default="bert", description="Model: bilstm|bert|ensemble"),
    text: str | None = Query(default=None),
    registry: ModelRegistry = Depends(get_model_registry),
    _api_key: str | None = Depends(get_api_key),
):
    """SSE proksi pipeline generator: preprocess 30 → ocr 55 → classify 80 → mapping 90 → done 100.

    Sumber input: ?text=... langsung, atau ?jobId=... (bytes dibaca dari disk;
    bila tidak ada, emit error). Heartbeat berupa komentar ':ping' tiap 15 dtk.
    """
    req_model = (model or "bert").lower()
    if req_model not in ("bilstm", "bert", "ensemble"):
        req_model = "bert"
    req_text = (text or "").strip() or None
    req_job = (jobId or job_id or "").strip() or None

    async def gen():
        loop = asyncio.get_running_loop()
        last_beat = time.monotonic()
        try:
            ready = (
                registry.is_ready("bert")
                if req_model == "bert"
                else registry.is_ready("ensemble")
                if req_model == "ensemble"
                else registry.is_loaded
            )
            if not ready:
                detail = (
                    "Model BERT belum dimuat" if req_model == "bert"
                    else "Ensemble butuh BiLSTM+BERT; salah satu belum dimuat"
                    if req_model == "ensemble"
                    else "Models not loaded"
                )
                yield _sse(_error_event(detail, 0, {"model": req_model}))
                return

            pipeline = DetectionPipeline(registry)
            if req_text:
                iterator = pipeline.detect_from_text_stream(req_text, model=req_model)
            else:
                if not req_job:
                    yield _sse(_error_event("jobId atau text wajib diisi", 0, {}))
                    return
                blob = await loop.run_in_executor(None, _find_job_bytes, req_job)
                if not blob:
                    yield _sse(_error_event(
                        f"Job {req_job} tidak ditemukan", 0, {"jobId": req_job}))
                    return
                iterator = pipeline.detect_from_image_stream(blob, model=req_model)

            it = iter(iterator)
            while True:
                pending = loop.run_in_executor(None, next, it)
                try:
                    while not pending.done():
                        await asyncio.sleep(0.25)
                        if time.monotonic() - last_beat >= HEARTBEAT_INTERVAL_S:
                            yield ":ping\n\n"
                            last_beat = time.monotonic()
                    try:
                        event = pending.result()
                    except StopIteration as stop:
                        result = stop.value
                        done = {
                            "stage": "done", "progress": 100,
                            "message": "Deteksi selesai",
                            "data": {"result": result_to_dict(result)},
                        }
                        yield _sse(done)
                        break
                    yield _sse(event)
                    last_beat = time.monotonic()
                finally:
                    if not pending.done():
                        pending.cancel()
        except ValueError as e:
            yield _sse(_error_event(str(e) or "Input tidak valid", 0, {"model": req_model}))
        except RuntimeError as e:
            yield _sse(_error_event(str(e) or "Models not loaded", 0, {"model": req_model}))
        except Exception as e:
            logger.error(f"Stream detection failed: {e}")
            yield _sse(_error_event("Detection failed", 0, {"model": req_model}))

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/detection/{detection_id}")
async def get_detection_result(detection_id: str):
    raise HTTPException(status_code=501, detail="Result storage not implemented")
