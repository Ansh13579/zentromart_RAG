import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
GNANI_API_KEY = os.getenv("GNANI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
IFM_API_KEY = os.getenv("IFM_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "ifm" if IFM_API_KEY else "gemini" if GEMINI_API_KEY else "nvidia",
).strip().casefold()

GNANI_STT_LANGUAGE = os.getenv("GNANI_STT_LANGUAGE", "en-IN")
GNANI_TTS_LANGUAGE = os.getenv("GNANI_TTS_LANGUAGE", GNANI_STT_LANGUAGE)
GNANI_TTS_MODEL = os.getenv("GNANI_TTS_MODEL", "timbre-v2.5")
GNANI_TTS_VOICE = os.getenv("GNANI_TTS_VOICE", "Pranav")
GNANI_MAX_AUDIO_DURATION_SECONDS = float(
    os.getenv("GNANI_MAX_AUDIO_DURATION_SECONDS", "60")
)
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").strip().casefold() in {
    "1", "true", "yes", "on"
}
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
).strip()
RERANKER_CANDIDATE_COUNT = max(
    5,
    int(os.getenv("RERANKER_CANDIDATE_COUNT", "25")),
)
RERANKER_BATCH_SIZE = max(1, int(os.getenv("RERANKER_BATCH_SIZE", "4")))
RERANKER_MAX_LENGTH = max(128, int(os.getenv("RERANKER_MAX_LENGTH", "384")))
VOICE_OUTPUT_DIR = Path(
    os.getenv("VOICE_OUTPUT_DIR", str(PROJECT_ROOT / "sample_outputs"))
)
