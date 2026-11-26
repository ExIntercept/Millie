"""Central configuration, read from environment (.env supported)."""
import os
import platform
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("MILLIE_DATA_DIR", ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "millie.db"
FRONTEND_DIR = ROOT / "frontend"

ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Millie")
TIMEZONE = os.getenv("TIMEZONE", "Asia/Kolkata")

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-flash-latest")
LLM_STREAM = os.getenv("LLM_STREAM", "1") == "1"

WAKEWORD_MODEL = os.getenv("WAKEWORD_MODEL", "")
WAKEWORD_THRESHOLD = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))
STT_MODEL = os.getenv("STT_MODEL", "mlx-community/whisper-small-mlx")
TTS_BACKEND = os.getenv("TTS_BACKEND", "say" if platform.system() == "Darwin" else "browser")
TTS_VOICE = os.getenv("TTS_VOICE", "Samantha")

SAMPLE_RATE = 16000
END_SILENCE_MS = int(os.getenv("END_SILENCE_MS", "900"))        # silence that ends your command
NO_SPEECH_TIMEOUT_S = 6.0   # give up if you say nothing after the wake word
MAX_UTTERANCE_S = 30.0
RESEARCH_MAX_STEPS = 8

WORKSPACE_DIR = DATA_DIR / "workspace"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en")
KOKORO_MODEL = os.getenv("KOKORO_MODEL", str(DATA_DIR / "kokoro-v1.0.onnx"))
KOKORO_VOICES = os.getenv("KOKORO_VOICES", str(DATA_DIR / "voices-v1.0.bin"))
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_sarah")
