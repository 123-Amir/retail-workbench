"""Central settings. Secrets come only from environment variables."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data/runtime"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
    max_result_rows: int = int(os.getenv("MAX_RESULT_ROWS", "200"))
    query_timeout: float = float(os.getenv("QUERY_TIMEOUT_SECONDS", "20"))


settings = Settings()
