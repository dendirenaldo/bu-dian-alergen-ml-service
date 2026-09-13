from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    MODEL_DIR: str = "./models"
    VOCAB_SIZE: int = 20000
    EMBED_DIM: int = 100
    MAX_LEN: int = 120
    TESSERACT_LANG: str = "ind+eng"
    OCR_MIN_CONF: int = 35
    BATCH_SIZE: int = 64
    EPOCHS: int = 150
    LEARNING_RATE: float = 1e-4
    TEST_SIZE: float = 0.2
    CSV_DELIMITER: str = ";"
    DEFAULT_THRESHOLD: float = 0.5
    # --- BERT / ensemble (dual-model) ---
    BERT_MODEL_DIR: str = "./models/bert"
    BERT_MODEL_NAME: str = "indobenchmark/indobert-base-p1"
    BERT_MAX_LEN: int = 256
    BERT_THRESHOLD: float = 0.5
    # 'bilstm' | 'bert' | 'ensemble' — default ensemble bila BERT tersedia.
    DEFAULT_MODEL: str = "bilstm"
    ENSEMBLE_STRATEGY: str = "weighted"  # 'weighted' | 'vote'
    ENSEMBLE_WEIGHT_BILSTM: float = 0.4
    ENSEMBLE_WEIGHT_BERT: float = 0.6
    NESTJS_API_URL: str = "http://localhost:3001"
    ML_API_KEY: str = ""
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    ALLOWED_DATA_DIRS: str = "./data"
    MIN_WORD_COUNT: int = 3
    W2V_WINDOW: int = 5
    W2V_EPOCHS: int = 30

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
