import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _req(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required env var {name}")
    return v


PG_DSN = (
    f"host={_req('POSTGRES_HOST', 'localhost')} port={_req('POSTGRES_PORT', '5432')} "
    f"dbname={_req('POSTGRES_DB')} user={_req('POSTGRES_USER')} password={_req('POSTGRES_PASSWORD')}"
)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
LLM_API_KEY = OPENROUTER_API_KEY or os.getenv("MISTRAL_API_KEY", "").strip()
LLM_URL = os.getenv("LLM_URL") or os.getenv("MISTRAL_URL") or (
    "https://openrouter.ai/api/v1/chat/completions" if OPENROUTER_API_KEY else "https://api.mistral.ai/v1/chat/completions"
)
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("MISTRAL_MODEL") or (
    "mistralai/mistral-small-24b-instruct-2501" if OPENROUTER_API_KEY else "open-mistral-7b"
)

MISTRAL_API_KEY = LLM_API_KEY
MISTRAL_MODEL = LLM_MODEL
MISTRAL_URL = LLM_URL
MISTRAL_TEMPERATURE = float(os.getenv("MISTRAL_TEMPERATURE", "0.2"))
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")
REASONS_MAX = int(os.getenv("REASONS_MAX", "400"))
REASONS_CONCURRENCY = int(os.getenv("REASONS_CONCURRENCY", "3"))
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", "400000"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
SEED = int(os.getenv("SEED", "42"))

MODEL_VERSION = "1.0.0"


def current_week() -> str:
    y, w, _ = date.today().isocalendar()
    return f"{y}-W{w:02d}"
