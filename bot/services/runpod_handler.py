"""
RunPod Handler — ComfyUI Serverless Flux image generation.

Works with the ComfyUI RunPod template which uses FLUX.1-dev-fp8.
Completely uncensored — no safety checker, no filters.

Supports:
- Text-to-Image via dynamic ComfyUI workflow JSON
- Image-to-Image (Magic Edit) via workflow with LoadImage + VAEEncode
- Synchronous (runsync) with timeout + async fallback (run + poll)
- Base64 image decoding from ComfyUI output
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
import time
from pathlib import Path
from typing import Any

import aiohttp

from config import (
    RUNPOD_API_KEY,
    RUNPOD_ENDPOINT_ID,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    IMG2IMG_STRENGTH,
    TEMP_DIR,
    logger,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUNPOD_BASE = "https://api.runpod.ai/v2"

# Timeouts (seconds)
RUNPOD_SYNC_TIMEOUT = 180       # runsync may take up to 3 min on cold start
RUNPOD_POLL_INTERVAL = 2.0      # poll every 2 seconds
RUNPOD_MAX_WAIT = 300           # max 5 minutes total wait
RUNPOD_FALLBACK_WAIT = 90       # shorter wait for fallback mode

# Retry settings
RUNPOD_MAX_RETRIES = 2          # retry on transient failures
RUNPOD_RETRY_DELAY = 3.0        # seconds between retries

# HTTP headers
_HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}


# ---------------------------------------------------------------------------
# ComfyUI Workflow Builder
# ---------------------------------------------------------------------------

# Base workflow template — same node IDs as the ComfyUI template example.
# We inject: prompt, negative_prompt, width, height, seed, steps, cfg_scale.
# Node ID mapping:
#   6  = CLIPTextEncode (positive prompt)
#   8  = VAEDecode
#   9  = SaveImage
#   27 = EmptySD3LatentImage (width, height, batch_size)
#   30 = CheckpointLoaderSimple
#   31 = KSampler (seed, steps, cfg, sampler, scheduler, denoise)
#   33 = CLIPTextEncode (negative prompt)
#   35 = FluxGuidance
#   38 = PreviewImage
#   40 = SaveImage (duplicate for output)

BASE_WORKFLOW: dict[str, Any] = {
    "6": {
        "inputs": {
            "text": "__PROMPT__",
            "clip": ["30", 1],
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
    },
    "8": {
        "inputs": {
            "samples": ["31", 0],
            "vae": ["30", 2],
        },
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"},
    },
    "9": {
        "inputs": {
            "filename_prefix": "ComfyUI",
            "images": ["8", 0],
        },
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
    },
    "27": {
        "inputs": {
            "width": DEFAULT_WIDTH,
            "height": DEFAULT_HEIGHT,
            "batch_size": 1,
        },
        "class_type": "EmptySD3LatentImage",
        "_meta": {"title": "EmptySD3LatentImage"},
    },
    "30": {
        "inputs": {
            "ckpt_name": "flux1-dev-fp8.safetensors",
        },
        "class_type": "CheckpointLoaderSimple",
        "_meta": {"title": "Load Checkpoint"},
    },
    "31": {
        "inputs": {
            "seed": 42,
            "steps": DEFAULT_NUM_STEPS,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "model": ["30", 0],
            "positive": ["35", 0],
            "negative": ["33", 0],
            "latent_image": ["27", 0],
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"},
    },
    "33": {
        "inputs": {
            "text": "__NEGATIVE__",
            "clip": ["30", 1],
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
    },
    "35": {
        "inputs": {
            "guidance": DEFAULT_GUIDANCE_SCALE,
            "conditioning": ["6", 0],
        },
        "class_type": "FluxGuidance",
        "_meta": {"title": "FluxGuidance"},
    },
    "38": {
        "inputs": {
            "images": ["8", 0],
        },
        "class_type": "PreviewImage",
        "_meta": {"title": "Preview Image"},
    },
    "40": {
        "inputs": {
            "filename_prefix": "ComfyUI",
            "images": ["8", 0],
        },
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
    },
}

# Default negative prompt — keeps quality high, avoids common artifacts
DEFAULT_NEGATIVE = (
    "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
    "watermark, text, signature, extra fingers, fused fingers, "
    "poorly drawn, out of frame, disfigured"
)


def _build_workflow(
    prompt: str,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    num_inference_steps: int = DEFAULT_NUM_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    negative_prompt: str = "",
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Build a ComfyUI workflow JSON with the user's parameters injected.

    Returns a deep copy of the base workflow with all values substituted.
    """
    workflow = json.loads(json.dumps(BASE_WORKFLOW))  # deep copy

    if seed is None:
        seed = random.randint(1, 2**63 - 1)

    neg = negative_prompt.strip() if negative_prompt else DEFAULT_NEGATIVE

    # Inject values into the workflow nodes
    workflow["6"]["inputs"]["text"] = prompt
    workflow["27"]["inputs"]["width"] = width
    workflow["27"]["inputs"]["height"] = height
    workflow["31"]["inputs"]["seed"] = seed
    workflow["31"]["inputs"]["steps"] = num_inference_steps
    workflow["31"]["inputs"]["cfg"] = 1.0  # Flux always uses cfg=1.0
    workflow["33"]["inputs"]["text"] = neg
    workflow["35"]["inputs"]["guidance"] = guidance_scale

    return workflow


def _build_img2img_workflow(
    prompt: str,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    num_inference_steps: int = DEFAULT_NUM_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    strength: float = IMG2IMG_STRENGTH,
    negative_prompt: str = "",
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Build a ComfyUI workflow for Image-to-Image.

    Adds LoadImage (node 50) + VAEEncode (node 51) before the KSampler.
    The KSampler's denoise is set to `strength` and latent_image comes from VAEEncode.
    """
    workflow = _build_workflow(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        seed=seed,
    )

    # Add LoadImage node (referenced by name "input_image.png" from images array)
    workflow["50"] = {
        "inputs": {
            "image": "input_image.png",
        },
        "class_type": "LoadImage",
        "_meta": {"title": "Load Image"},
    }

    # Add VAEEncode node — encodes the loaded image into latent space
    workflow["51"] = {
        "inputs": {
            "pixels": ["50", 0],
            "vae": ["30", 2],
        },
        "class_type": "VAEEncode",
        "_meta": {"title": "VAE Encode"},
    }

    # Redirect KSampler latent_image to the encoded image
    workflow["31"]["inputs"]["latent_image"] = ["51", 0]
    # Set denoise = strength (how much to change)
    workflow["31"]["inputs"]["denoise"] = strength

    return workflow


# ---------------------------------------------------------------------------
# Image download helper
# ---------------------------------------------------------------------------

async def _download_image(url: str, save_path: Path) -> Path:
    """Download an image from a URL to a local file."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(
                    f"Image download failed: HTTP {resp.status} — {text[:200]}"
                )
            data = await resp.read()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(data)
    logger.info(f"Image saved: {save_path} ({len(data)} bytes)")
    return save_path


# ---------------------------------------------------------------------------
# Base64 decoding helper
# ---------------------------------------------------------------------------

def _decode_base64_image(data_uri: str, save_to: Path) -> Path:
    """Decode a data URI like 'data:image/png;base64,...' and save to disk."""
    if "," in data_uri:
        _, b64_part = data_uri.split(",", 1)
    else:
        b64_part = data_uri

    raw = base64.b64decode(b64_part)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    save_to.write_bytes(raw)
    logger.info(f"Base64 image decoded and saved: {save_to} ({len(raw)} bytes)")
    return save_to


# ---------------------------------------------------------------------------
# Extract image from ComfyUI output
# ---------------------------------------------------------------------------

def _extract_image_from_output(output: Any) -> bytes | None:
    """
    Extract image bytes from ComfyUI RunPod output regardless of the varying structures.
    """
    if not output:
        return None

    def _decode(val: Any) -> bytes | None:
        if isinstance(val, str):
            if val.startswith("data:image"):
                _, val = val.split(",", 1)
            elif val.startswith("http"):
                return None  # Download not handled here
            try:
                return base64.b64decode(val)
            except Exception as e:
                logger.error(f"Base64 decode error: {e}")
                return None
        return None

    if isinstance(output, dict):
        # 1. output.message
        msg = output.get("message")
        if isinstance(msg, str) and msg.startswith("data:image"):
            return _decode(msg)
        if isinstance(msg, list) and len(msg) > 0:
            first = msg[0]
            if isinstance(first, dict):
                val = first.get("image") or first.get("data")
                if val:
                    return _decode(val)

        # 2. output.images
        imgs = output.get("images")
        if isinstance(imgs, list) and len(imgs) > 0:
            first = imgs[0]
            if isinstance(first, dict):
                val = first.get("image") or first.get("data")
                if val:
                    return _decode(val)
            elif isinstance(first, str):
                return _decode(first)

        # 3. output.image or output.image_url
        img_str = output.get("image") or output.get("image_url")
        if isinstance(img_str, str):
            return _decode(img_str)

    # 4. output as list
    if isinstance(output, list) and len(output) > 0:
        first = output[0]
        if isinstance(first, str):
            return _decode(first)
        if isinstance(first, dict):
            return _extract_image_from_output(first)

    # 5. output as plain string
    if isinstance(output, str):
        return _decode(output)

    return None


# ---------------------------------------------------------------------------
# Polling helper (async mode)
# ---------------------------------------------------------------------------

async def _poll_runpod_job(
    job_id: str,
    poll_interval: float = RUNPOD_POLL_INTERVAL,
    max_wait: float = RUNPOD_MAX_WAIT,
) -> dict[str, Any]:
    """Poll a RunPod async job until completion or timeout."""
    url = f"{RUNPOD_BASE}/{RUNPOD_ENDPOINT_ID}/status/{job_id}"
    start = time.monotonic()

    async with aiohttp.ClientSession() as session:
        while True:
            elapsed = time.monotonic() - start
            if elapsed > max_wait:
                raise TimeoutError(
                    f"RunPod job {job_id} timed out after {max_wait:.0f}s"
                )

            async with session.get(
                url, headers=_HEADERS, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning(f"Poll status {resp.status}: {text[:150]}")
                    await asyncio.sleep(poll_interval)
                    continue

                data: dict[str, Any] = await resp.json()

            status = data.get("status", "").lower()

            if status == "completed":
                logger.info(f"RunPod job {job_id} completed in {elapsed:.1f}s")
                return data
            elif status == "failed":
                error_msg = data.get("error", "Unknown error")
                raise RuntimeError(f"RunPod job {job_id} failed: {error_msg}")
            elif status == "cancelled":
                raise RuntimeError(f"RunPod job {job_id} was cancelled")
            elif status in ("in_progress", "in_queue"):
                delay = data.get("delayTime", poll_interval * 1000) / 1000.0
                sleep_for = max(poll_interval, min(delay, 10.0))
                logger.debug(
                    f"Job {job_id}: {status} ({elapsed:.0f}s elapsed), "
                    f"polling again in {sleep_for:.1f}s"
                )
                await asyncio.sleep(sleep_for)
            else:
                logger.warning(f"Unknown RunPod status '{status}' — retrying")
                await asyncio.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Core: call RunPod with ComfyUI workflow
# ---------------------------------------------------------------------------

async def _runpod_comfyui_request(
    workflow: dict[str, Any],
    images: list[dict[str, str]] | None = None,
) -> bytes | None:
    """
    Send a ComfyUI workflow to RunPod.

    Strategy:
    1. Try runsync (single HTTP call) — best for warm workers.
    2. If runsync times out, fall back to async run + poll.

    Returns raw image bytes or None.
    """
    payload: dict[str, Any] = {"workflow": workflow}
    if images:
        payload["images"] = images

    for attempt in range(RUNPOD_MAX_RETRIES + 1):
        try:
            # --- Attempt 1: Synchronous ---
            sync_url = f"{RUNPOD_BASE}/{RUNPOD_ENDPOINT_ID}/runsync"
            timeout = aiohttp.ClientTimeout(total=RUNPOD_SYNC_TIMEOUT)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    sync_url,
                    json={"input": payload},
                    headers=_HEADERS,
                    timeout=timeout,
                ) as resp:
                    data: dict[str, Any] = await resp.json()

            status = data.get("status", "").lower()

            if status == "completed":
                output = data.get("output", {})
                image_bytes = _extract_image_from_output(output)
                if image_bytes:
                    logger.info(
                        f"RunPod runsync OK — {len(image_bytes)} bytes "
                        f"in {data.get('executionTime', '?')}ms"
                    )
                    return image_bytes

                logger.warning("RunPod runsync completed but no image in output")
                return None

            elif status == "failed":
                error_msg = data.get("error", "Unknown")
                raise RuntimeError(f"RunPod runsync failed: {error_msg}")

            elif status in ("in_progress", "in_queue"):
                job_id = data.get("id", "")
                if job_id:
                    logger.info(
                        f"RunPod runsync returned {status}, "
                        f"falling back to poll job {job_id}"
                    )
                    poll_result = await _poll_runpod_job(
                        job_id,
                        max_wait=RUNPOD_FALLBACK_WAIT,
                    )
                    output = poll_result.get("output", {})
                    return _extract_image_from_output(output)
                else:
                    raise RuntimeError(f"RunPod {status} but no job ID returned")

            else:
                logger.warning(
                    f"Unexpected runsync status: {status}, retrying..."
                )
                if attempt < RUNPOD_MAX_RETRIES:
                    await asyncio.sleep(RUNPOD_RETRY_DELAY)
                    continue
                return None

        except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
            logger.warning(
                f"RunPod runsync attempt {attempt+1} failed ({exc}), "
                f"will try async fallback"
            )
            return await _runpod_async(payload)

        except RuntimeError:
            raise

        except Exception as exc:
            logger.error(f"RunPod unexpected error (attempt {attempt+1}): {exc}")
            if attempt < RUNPOD_MAX_RETRIES:
                await asyncio.sleep(RUNPOD_RETRY_DELAY)
            else:
                raise

    return None


# ---------------------------------------------------------------------------
# Async-only mode (used as fallback)
# ---------------------------------------------------------------------------

async def _runpod_async(payload: dict[str, Any]) -> bytes | None:
    """Submit async job + poll until complete."""
    run_url = f"{RUNPOD_BASE}/{RUNPOD_ENDPOINT_ID}/run"

    async with aiohttp.ClientSession() as session:
        async with session.post(
            run_url,
            json={"input": payload},
            headers=_HEADERS,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(
                    f"RunPod async run failed: HTTP {resp.status} — {text[:200]}"
                )
            data: dict[str, Any] = await resp.json()

    job_id = data.get("id", "")
    if not job_id:
        raise RuntimeError("RunPod async run returned no job ID")

    logger.info(f"RunPod async job submitted: {job_id}")
    poll_result = await _poll_runpod_job(job_id)
    output = poll_result.get("output", {})
    return _extract_image_from_output(output)


# ---------------------------------------------------------------------------
# Public API — Text-to-Image
# ---------------------------------------------------------------------------

async def call_runpod_text2img(
    prompt: str,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    num_inference_steps: int = DEFAULT_NUM_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    negative_prompt: str = "",
    seed: int | None = None,
    save_to: Path | None = None,
) -> Path:
    """
    Generate an image from a text prompt using ComfyUI + Flux on RunPod.

    Args:
        prompt: English prompt (already translated + expanded by Gemini).
        width: Image width (default 1024).
        height: Image height (default 1024).
        num_inference_steps: Denoising steps (default 28).
        guidance_scale: FluxGuidance scale (default 3.5).
        negative_prompt: What to avoid in the image.
        seed: Optional seed for reproducibility.
        save_to: Path to save the image. Auto-generated if None.

    Returns:
        Path to the saved image file.

    Raises:
        RuntimeError: If generation fails after all retries.
        TimeoutError: If RunPod takes too long.
        ValueError: If prompt is empty.
    """
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    logger.info(
        f"Text-to-Image request: {width}x{height}, "
        f"steps={num_inference_steps}, cfg={guidance_scale}"
    )
    logger.debug(f"Prompt: {prompt[:150]}...")

    # Build the ComfyUI workflow
    workflow = _build_workflow(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        seed=seed,
    )

    # Send to RunPod
    image_bytes = await _runpod_comfyui_request(workflow)

    if not image_bytes:
        raise RuntimeError("RunPod returned no image data")

    # Save to disk
    if save_to is None:
        save_to = TEMP_DIR / f"flux_txt2img_{int(time.time())}.png"

    save_to.parent.mkdir(parents=True, exist_ok=True)
    save_to.write_bytes(image_bytes)
    logger.info(f"Text-to-Image saved: {save_to} ({len(image_bytes)} bytes)")
    return save_to


# ---------------------------------------------------------------------------
# Public API — Image-to-Image (Magic Edit)
# ---------------------------------------------------------------------------

async def call_runpod_img2img(
    prompt: str,
    image_bytes: bytes,
    *,
    strength: float = IMG2IMG_STRENGTH,
    num_inference_steps: int = DEFAULT_NUM_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    negative_prompt: str = "",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    seed: int | None = None,
    save_to: Path | None = None,
) -> Path:
    """
    Edit an existing image using ComfyUI + Flux Image-to-Image.

    Args:
        prompt: Edit instruction (translated to English).
        image_bytes: Raw bytes of the source image.
        strength: How much to change (0.0 = identical, 1.0 = completely new).
        num_inference_steps: Denoising steps.
        guidance_scale: FluxGuidance scale.
        negative_prompt: What to avoid.
        width: Target width.
        height: Target height.
        seed: Optional seed.
        save_to: Save path (auto-generated if None).

    Returns:
        Path to the edited image file.
    """
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not image_bytes:
        raise ValueError("image_bytes must not be empty")

    logger.info(
        f"Image-to-Image request: {width}x{height}, "
        f"strength={strength}, steps={num_inference_steps}"
    )
    logger.debug(f"Edit instruction: {prompt[:150]}...")

    # Build the img2img workflow
    workflow = _build_img2img_workflow(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        negative_prompt=negative_prompt,
        seed=seed,
    )

    # Encode source image as base64 for the ComfyUI LoadImage node
    encoded = base64.b64encode(image_bytes).decode("ascii")
    images_payload = [
        {
            "name": "input_image.png",
            "image": encoded,
        }
    ]

    # Send to RunPod
    result_bytes = await _runpod_comfyui_request(workflow, images=images_payload)

    if not result_bytes:
        raise RuntimeError("RunPod img2img returned no image data")

    if save_to is None:
        save_to = TEMP_DIR / f"flux_img2img_{int(time.time())}.png"

    save_to.parent.mkdir(parents=True, exist_ok=True)
    save_to.write_bytes(result_bytes)
    logger.info(f"Image-to-Image saved: {save_to} ({len(result_bytes)} bytes)")
    return save_to


# ---------------------------------------------------------------------------
# Quick health check
# ---------------------------------------------------------------------------

async def check_runpod_health() -> dict[str, Any] | None:
    """
    Check if the RunPod endpoint is reachable and has active workers.
    Returns endpoint info dict or None if unreachable.
    """
    url = f"{RUNPOD_BASE}/{RUNPOD_ENDPOINT_ID}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=_HEADERS,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"RunPod health: {data}")
                    return data
                else:
                    text = await resp.text()
                    logger.warning(
                        f"RunPod health check failed: {resp.status} — {text[:200]}"
                    )
                    return None
    except Exception as exc:
        logger.warning(f"RunPod health check error: {exc}")
        return None