"""
AI Image Bot - Configuration Management
Loads settings from .env file and provides typed access to all configuration.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN is not set in .env file")

ADMIN_USER_ID: int | None = None
_raw_admin = os.getenv("ADMIN_USER_ID", "")
if _raw_admin.strip():
    ADMIN_USER_ID = int(_raw_admin)


# ---------------------------------------------------------------------------
# Gemini (Google AI)
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env file")

GEMINI_MODEL: str = "gemini-3.5-flash"

# Rate limiting: free tier = 15 RPM, 1500 RPD
GEMINI_RPM_LIMIT: int = 14  # leave 1 buffer
GEMINI_RPD_LIMIT: int = 1400  # leave buffer


# ---------------------------------------------------------------------------
# RunPod Serverless
# ---------------------------------------------------------------------------
RUNPOD_API_KEY: str = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID: str = os.getenv("RUNPOD_ENDPOINT_ID", "")

# Don't crash at startup — just warn. User will add creds later.
_RUNPOD_READY: bool = (
    bool(RUNPOD_API_KEY)
    and bool(RUNPOD_ENDPOINT_ID)
    and "YOUR_" not in RUNPOD_API_KEY
    and "YOUR_" not in RUNPOD_ENDPOINT_ID
)

# Image generation defaults
DEFAULT_WIDTH: int = 1024
DEFAULT_HEIGHT: int = 1024
DEFAULT_NUM_STEPS: int = 28
DEFAULT_GUIDANCE_SCALE: float = 3.5

# Image-to-Image (modify mode)
IMG2IMG_STRENGTH: float = 0.4  # 0.3-0.5 range, lower = closer to original


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH: Path = BASE_DIR / "data" / "bot.db"

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
FONTS_DIR: Path = BASE_DIR / "fonts"
DEFAULT_ARABIC_FONT: str = "NotoNaskhArabic-Regular.ttf"
DEFAULT_ARABIC_FONT_BOLD: str = "NotoNaskhArabic-Bold.ttf"
FONT_SIZE_DEFAULT: int = 48
FONT_COLOR: str = "white"
FONT_OUTLINE_COLOR: str = "black"
FONT_OUTLINE_WIDTH: int = 2

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("ai_image_bot")

# Warn about missing RunPod credentials now that logger is set up
if not _RUNPOD_READY:
    logger.warning(
        "RunPod credentials not configured — image generation will be "
        "unavailable until RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID are set in .env"
    )

# ---------------------------------------------------------------------------
# Paths – ensure directories exist
# ---------------------------------------------------------------------------
DATA_DIR: Path = BASE_DIR / "data"
TEMP_DIR: Path = BASE_DIR / "temp_images"

for _dir in (DATA_DIR, TEMP_DIR):
    _dir.mkdir(parents=True, exist_ok=True)