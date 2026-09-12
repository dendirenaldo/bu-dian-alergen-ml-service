# Bu Dian ML Service

API inferensi untuk deteksi alergen makanan menggunakan pipeline OCR + Word2Vec + BiLSTM. Dibangun dengan FastAPI.

## Tech Stack

- Python 3.11
- FastAPI + Uvicorn
- TensorFlow / Keras 2.17
- Gensim (Word2Vec)
- OpenCV (image preprocessing)
- Tesseract OCR (ekstraksi teks)
- Sastrawi (NLP Indonesia)
- Pydantic (data validation)

## Pipeline Deteksi

```
Gambar Makanan
    │
    ▼
Preprocessing Gambar (OpenCV) — grayscale, thresholding, denoising
    │
    ▼
OCR - Ekstraksi Teks (Tesseract) — deteksi teks komposisi
    │
    ▼
Parsing Teks Komposisi — ekstraksi bahan dari label
    │
    ▼
Preprocessing Teks (Sastrawi) — stopword removal, normalisasi
    │
    ▼
Word2Vec Embedding — vektorisasi teks
    │
    ▼
Klasifikasi BiLSTM — prediksi safe/unsafe
    │
    ▼
Hasil: { result: "safe"|"unsafe", confidence_score, allergens[] }
```

## Setup

### 1. Buat virtual environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# atau
venv\Scripts\activate     # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download installer dari https://github.com/UB-Mannheim/tesseract/wiki
```

### 4. Konfigurasi environment

```bash
cp .env.example .env
```

```env
NESTJS_API_URL=http://localhost:3001
ML_API_KEY=your-ml-api-key
MODEL_DIR=./models
API_V1_PREFIX=/api/v1
DEBUG=false
```

### 5. Jalankan server

```bash
# Development (auto-reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Server berjalan di http://localhost:8000

## API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | Root message |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/health/ready` | Readiness check (model loaded?) |
| POST | `/api/v1/detection/upload` | Upload gambar untuk deteksi |
| POST | `/api/v1/detection/text` | Klasifikasi teks komposisi |
| GET | `/api/v1/detection/{id}` | Ambil hasil deteksi |
| GET | `/api/v1/model/status` | Status model |
| GET | `/api/v1/model/metrics` | Metrik model |
| POST | `/api/v1/model/train` | Trigger training ulang |
| GET | `/api/v1/model/train/status` | Status training |

## Model Files

Letakkan file model yang sudah di-training di direktori `models/`:

```
models/
├── bilstm_model.keras    # Model BiLSTM
├── word2vec.model        # Model Word2Vec (Gensim)
├── tokenizer.pkl         # Tokenizer (Keras)
└── label_encoder.pkl     # Label encoder
```

## Docker

```bash
# Build image
docker build -t budian-ml-service .

# Jalankan container
docker run -p 8000:8000 -e ML_API_KEY=your-key budian-ml-service
```

## Interactive API Docs

Swagger UI: http://localhost:8000/docs

ReDoc: http://localhost:8000/redoc
