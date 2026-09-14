"""
Backend configuration.

Values are read from environment variables (or a .env file loaded by
python-dotenv at startup).  Do not import this module at the package
level in retrieval.py or ingestion.py; those modules receive
configuration values as constructor arguments so they stay testable
without environment setup.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env when present; silently skip when absent (e.g. in tests).
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)


def normalize_optional_path_environment(name: str) -> str:
    """Treat a blank optional path as unset instead of the current directory."""
    value = os.getenv(name, "").strip()
    if not value:
        os.environ.pop(name, None)
        return ""
    return value


_HF_HOME = normalize_optional_path_environment("HF_HOME")


class _Config:
    # ------------------------------------------------------------------ #
    # Credentials                                                          #
    # ------------------------------------------------------------------ #
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

    # ------------------------------------------------------------------ #
    # Runtime                                                              #
    # ------------------------------------------------------------------ #
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    backend_url: str = os.getenv("BACKEND_URL", "http://localhost:8000")
    # Optional redirect for Hugging Face model cache.
    hf_home: str = _HF_HOME

    # ------------------------------------------------------------------ #
    # Embedding / retrieval                                                #
    # ------------------------------------------------------------------ #
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # BGE asymmetric retrieval instruction applied to queries only.
    bge_query_prefix: str = "Represent this sentence for searching relevant passages: "
    # Locked from the recorded calibration run. Environment overrides remain
    # server-owned operational configuration, never request parameters.
    min_relevance_score: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.625"))
    top_k: int = int(os.getenv("TOP_K", "3"))

    # ------------------------------------------------------------------ #
    # Generation                                                           #
    # ------------------------------------------------------------------ #
    generation_model: str = "gemini-3.1-flash-lite"
    temperature: float = 0.1
    max_output_tokens: int = 400
    # Gemini provider timeout in seconds.
    generation_timeout: float = 30.0
    # Slightly longer frontend HTTP timeout.
    frontend_timeout: float = 35.0

    # ------------------------------------------------------------------ #
    # Policies directory (relative to repository root)                     #
    # ------------------------------------------------------------------ #
    policies_dir: str = "policies"


config = _Config()
