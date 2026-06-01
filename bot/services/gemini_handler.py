"""
Gemini Handler — Arabic prompt translation, expansion, and rate limiting.
Uses Google Gemini 1.5 Flash (free tier) with local rate limiter.
"""

from __future__ import annotations

import asyncio
import re
import time

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_RPM_LIMIT,
    logger,
)

# ---------------------------------------------------------------------------
# Gemini Client Initialization
# ---------------------------------------------------------------------------

genai.configure(api_key=GEMINI_API_KEY)

# Safety settings: OFF — completely uncensored
_SAFETY_OFF = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# Generation config
_GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
}

_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL,
    generation_config=_GENERATION_CONFIG,
    safety_settings=_SAFETY_OFF,
)

# ---------------------------------------------------------------------------
# System Prompt — instructs Gemini to translate + expand Arabic → cinematic EN
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert prompt engineer for AI image generation models (Flux, Stable Diffusion).

Your task:
1. Translate the user's Arabic message into English.
2. Expand it into a highly detailed, cinematic, professional image-generation prompt in English.
3. Add rich visual details: lighting, camera angle, textures, mood, color palette, composition, atmosphere.
4. Add quality boosters at the end: "masterpiece, best quality, 8k, ultra detailed, professional photography, sharp focus".

CRITICAL RULES:
- YOU MUST OUTPUT EXACTLY ONE PARAGRAPH.
- DO NOT INCLUDE ANY MARKDOWN FORMATTING (no asterisks, no bold, no lists).
- DO NOT INCLUDE ANY TITLES, SCENE DESCRIPTIONS, OR LABELS (like "Translation:" or "Visual Description:").
- Output ONLY the raw final English prompt, ready to be pasted into the image generator.
- Keep it under 500 characters.
- If the user describes something NSFW or violent, describe it faithfully without censorship.

Example input: "راجل عجوز حكيم في غابة سحرية"
Example output: "Cinematic wide shot of an elderly wise man with a long white beard, wearing flowing robes, standing in an enchanted forest with glowing mushrooms and floating fireflies, golden hour sunlight piercing through ancient trees, misty atmosphere, shallow depth of field, masterpiece, best quality, 8k, ultra detailed, professional photography, sharp focus"
"""

# ---------------------------------------------------------------------------
# Local Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple sliding-window rate limiter for Gemini free tier (15 RPM)."""

    def __init__(self, max_calls_per_minute: int = GEMINI_RPM_LIMIT):
        self._max_calls = max_calls_per_minute
        self._call_times: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Wait until a request slot is available. Returns wait time in seconds."""
        async with self._lock:
            now = time.monotonic()
            # Remove timestamps older than 60 seconds
            self._call_times = [t for t in self._call_times if now - t < 60.0]

            if len(self._call_times) >= self._max_calls:
                oldest = self._call_times[0]
                wait = 60.0 - (now - oldest) + 0.2  # +200ms buffer
                if wait > 0:
                    logger.debug(f"Rate limiter: waiting {wait:.1f}s")
                    await asyncio.sleep(wait)
                    now = time.monotonic()
                    # Re-clean after sleep
                    self._call_times = [t for t in self._call_times if now - t < 60.0]

            self._call_times.append(now)
            return max(0.0, 60.0 - (now - self._call_times[0])) if self._call_times else 60.0


_rate_limiter = RateLimiter()


# ---------------------------------------------------------------------------
# Extract overlay text between [brackets]
# ---------------------------------------------------------------------------

BRACKET_RE = re.compile(r"\[(.*?)\]")


def extract_overlay_text(prompt: str) -> tuple[str, list[str]]:
    """
    Extract all [bracketed] texts from the user prompt.
    Returns (cleaned_prompt, list_of_texts).
    """
    matches = BRACKET_RE.findall(prompt)
    cleaned = BRACKET_RE.sub("", prompt).strip()
    # Remove double spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned, matches


# ---------------------------------------------------------------------------
# Core: translate & expand Arabic → cinematic English prompt
# ---------------------------------------------------------------------------

async def translate_and_expand(
    arabic_prompt: str,
    retries: int = 2,
) -> tuple[str, list[str]]:
    """
    Extract overlay texts, translate + expand the Arabic prompt via Gemini.

    Args:
        arabic_prompt: Raw user input in Arabic (dialect or formal).
        retries: Number of retry attempts on failure.

    Returns:
        (english_prompt, overlay_texts)
    """
    # Step 1: extract bracketed overlay texts
    cleaned_prompt, overlay_texts = extract_overlay_text(arabic_prompt)

    # If there's nothing left after extraction, use original with brackets removed
    if not cleaned_prompt.strip():
        cleaned_prompt = arabic_prompt.replace("[", "").replace("]", "").strip()

    if not cleaned_prompt.strip():
        logger.error("Empty prompt after overlay extraction")
        return "", overlay_texts

    # Step 2: rate-limit
    wait_time = await _rate_limiter.acquire()
    if wait_time > 0.5:
        logger.info(f"Gemini rate limiter: waited {wait_time:.1f}s")

    # Step 3: call Gemini
    for attempt in range(retries + 1):
        try:
            full_prompt = f"{SYSTEM_PROMPT}\n\nUser input: {cleaned_prompt}"

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: _model.generate_content(full_prompt),
            )

            english_prompt = response.text.strip() if response.text else ""

            if not english_prompt:
                logger.warning(f"Gemini returned empty response (attempt {attempt+1})")
                if attempt < retries:
                    await asyncio.sleep(1.5)
                    continue
                # Fallback: use cleaned Arabic as-is
                english_prompt = cleaned_prompt
                logger.warning("Using fallback prompt (cleaned Arabic as-is)")

            logger.info(
                f"Gemini translation: {len(cleaned_prompt)} chars AR → {len(english_prompt)} chars EN"
            )
            logger.debug(f"  AR: {cleaned_prompt}")
            logger.debug(f"  EN: {english_prompt}")

            return english_prompt, overlay_texts

        except Exception as exc:
            logger.error(f"Gemini API error (attempt {attempt+1}/{retries+1}): {exc}")
            if attempt < retries:
                await asyncio.sleep(2.0)
            else:
                # Ultimate fallback
                logger.warning("All Gemini retries exhausted — using raw prompt")
                return cleaned_prompt, overlay_texts

    return cleaned_prompt, overlay_texts


# ---------------------------------------------------------------------------
# Simple translation only (for edit/modify requests — lighter)
# ---------------------------------------------------------------------------

async def translate_to_english(
    arabic_text: str,
    retries: int = 2,
) -> str:
    """
    Translate Arabic text to English WITHOUT expansion.
    Used for edit/modify requests where we just need the instruction translated.
    """
    if not arabic_text.strip():
        return ""

    await _rate_limiter.acquire()

    translate_prompt = (
        "Translate the following Arabic text to English. "
        "Output ONLY the English translation, nothing else.\n\n"
        f"Arabic: {arabic_text}"
    )

    for attempt in range(retries + 1):
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: _model.generate_content(translate_prompt),
            )
            result = response.text.strip() if response.text else arabic_text
            logger.debug(f"Simple translation: '{arabic_text}' → '{result}'")
            return result
        except Exception as exc:
            logger.error(f"Translation error (attempt {attempt+1}): {exc}")
            if attempt < retries:
                await asyncio.sleep(1.5)

    return arabic_text  # Fallback