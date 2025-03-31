import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration"""

    # Flask settings
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-key-for-anomaly-detection-app")
    DEBUG = os.environ.get("DEBUG", "True").lower() == "true"

    # File upload settings
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
    MODEL_OUTPUT_FOLDER = os.environ.get("MODEL_OUTPUT_FOLDER", "model_output")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size

    # Default paths
    DEFAULT_CONFIG_PATH = os.environ.get("DEFAULT_CONFIG_PATH", "config/config.json")

    # LLM API settings
    LLM_API_URL = os.environ.get(
        "LLM_API_URL",
        "https://aiplatform.dev51.cbf.dev.paypalinc.com/seldon/seldon/mistral-7b-inst-624b0/v2/models/mistral-7b-inst-624b0/infer",
    )
    LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
    DEFAULT_MODEL_NAME = os.environ.get("DEFAULT_MODEL_NAME", "mistral-7b-inst-2252b")

    # Redis for Celery (if used)
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    # Database settings (if needed later)
    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///anomaly_detection.db")
