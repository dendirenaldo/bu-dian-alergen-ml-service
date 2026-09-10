# Bu Dian ML Service

FastAPI service untuk deteksi alergen makanan menggunakan OCR + BiLSTM.

## Tech Stack

- Python 3.11
- FastAPI
- TensorFlow / Keras
- Gensim (Word2Vec)
- OpenCV
- Tesseract OCR
- Sastrawi (Indonesian NLP)

## Pipeline

```
Image → Preprocessing (OpenCV) → OCR (Tesseract) → Composition Parsing
→ Text Preprocessing (Sastrawi) → Word2Vec Embedding → BiLSTM Classification
→ Safe / Unsafe
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env

# Run server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

```
POST   /api/v1/detection/upload   — Upload image for detection
POST   /api/v1/detection/text     — Classify ingredient text
POST   /api/v1/model/train        — Trigger model training
GET    /api/v1/model/status       — Check training status
GET    /api/v1/health             — Health check
```

## Model Files

Place trained model files in `models/` directory:
- `bilstm_model.keras` — BiLSTM model weights
- `word2vec.model` — Trained Word2Vec model
- `tokenizer.pkl` — Keras tokenizer
- `label_encoder.pkl` — Label encoder

## Docker

```bash
docker build -t budian-ml-service .
docker run -p 8000:8000 budian-ml-service
```
