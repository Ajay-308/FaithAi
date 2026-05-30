"""
Christian image generation module.

Architecture:
1. User prompt  →  moderation check
2. Prompt       →  gpt-4o-mini mediated prompt engineering (safe, reverent rewrite)
3. Rewritten prompt  →  image API
4. Return image bytes or URL

Image backends:
  Primary  → Stability AI v2beta (STABILITY_API_KEY required)
  Fallback → Pollinations (zero-config, demo/dev use)

FIXES:
  - generate_image_prompt now imported from prompt_rewriter (circular import fix)
  - _verify_url_accessible() is now actually called on Pollinations URL
  - Dead _verify_url_accessible that was never called is now used or clearly noted
"""

import os
import requests
import urllib.parse

from safety.moderation import moderate_image_prompt, RiskLevel
from services.prompt_rewriter import generate_image_prompt  # FIX: was from ai_engine (circular)


# ── Constants ─────────────────────────────────────────────────────────────────

STYLE_SUFFIX = (
    "classical Christian art style, Renaissance painting, warm golden light, "
    "reverent atmosphere, highly detailed, masterpiece quality"
)

POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}"

STABILITY_API_URL = "https://api.stability.ai/v2beta/stable-image/generate/core"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def generate_christian_image(user_request: str, denomination: str = "general") -> dict:
    """
    Full pipeline: moderate → rewrite → generate.

    Returns:
        {
          "success":      bool,
          "image_bytes":  bytes | None,   # present when Stability AI is used
          "image_url":    str | None,     # present when Pollinations fallback is used
          "prompt_used":  str | None,
          "reason":       str | None,
        }
    """
    # Step 1: Moderate raw request
    mod = moderate_image_prompt(user_request)
    if mod.level == RiskLevel.BLOCK:
        return {
            "success": False,
            "image_bytes": None,
            "image_url": None,
            "prompt_used": None,
            "reason": mod.suggested_response,
        }

    # Step 2: gpt-4o-mini rewrites into a safe, detailed, reverent art prompt
    try:
        safe_prompt = generate_image_prompt(user_request, denomination)
    except Exception:
        safe_prompt = f"Beautiful Christian artwork depicting {user_request}, {STYLE_SUFFIX}"

    # Step 3: Append style suffix if not already present
    final_prompt = safe_prompt
    keywords = ("renaissance", "classical", "byzantine", "painterly", "masterpiece")
    if not any(kw in safe_prompt.lower() for kw in keywords):
        final_prompt = f"{safe_prompt}, {STYLE_SUFFIX}"

    # Step 4: Try Stability AI first, then fall back to Pollinations
    stability_key = os.getenv("STABILITY_API_KEY")

    if stability_key:
        image_bytes = _generate_with_stability(final_prompt, stability_key)
        if image_bytes:
            return {
                "success": True,
                "image_bytes": image_bytes,
                "image_url": None,
                "prompt_used": final_prompt,
                "reason": None,
            }

    # Fallback to Pollinations
    image_url = _build_pollinations_url(final_prompt)

    # FIX: Actually validate the URL instead of silently ignoring _verify_url_accessible()
    if not _verify_url_accessible(image_url):
        return {
            "success": False,
            "image_bytes": None,
            "image_url": None,
            "prompt_used": final_prompt,
            "reason": "Image service is currently unavailable. Please try again later.",
        }

    return {
        "success": True,
        "image_bytes": None,
        "image_url": image_url,
        "prompt_used": final_prompt,
        "reason": None,
    }


# ── Stability AI backend (primary) ────────────────────────────────────────────

def _generate_with_stability(prompt: str, api_key: str) -> bytes | None:
    """
    Primary backend using Stability AI v2beta.
    Returns raw PNG image bytes on success, None on failure.
    """
    try:
        response = requests.post(
            STABILITY_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "image/*",
            },
            files={"none": ""},
            data={
                "prompt": prompt,
                "output_format": "png",
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.content

    except requests.HTTPError as e:
        print(f"Stability AI HTTP error: {e.response.status_code} – {e.response.text}")
        return None
    except Exception as e:
        print(f"Stability AI error: {e}")
        return None


# ── Pollinations backend (fallback) ───────────────────────────────────────────

def _build_pollinations_url(prompt: str) -> str:
    encoded = urllib.parse.quote(prompt)
    return POLLINATIONS_URL.format(prompt=encoded)


def _verify_url_accessible(url: str) -> bool:
    """
    Lightweight check: Pollinations generates images on GET, so HEAD may return
    404 or 405. We treat those as accessible and only fail on connection errors
    or clear server errors (5xx).
    """
    try:
        resp = requests.head(url, timeout=10, allow_redirects=True)
        # 4xx from Pollinations is normal (HEAD not supported) — still accessible
        return resp.status_code < 500
    except requests.RequestException:
        return False  # Genuine connectivity failure


# ── Optional: DALL-E 3 backend ────────────────────────────────────────────────

def _generate_with_dalle(prompt: str, api_key: str) -> str | None:
    """
    Production backend using OpenAI DALL-E 3.
    Returns image URL on success, None on failure.
    """
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return response.data[0].url
    except Exception as e:
        print(f"DALL-E 3 error: {e}")
        return None


# ── Example prompts ───────────────────────────────────────────────────────────

EXAMPLE_IMAGE_PROMPTS: list[str] = [
    "The Good Shepherd caring for his flock at golden hour",
    "The Sermon on the Mount, Jesus teaching on a hillside",
    "A dove descending over still water at dawn",
    "The empty tomb on Easter morning with light streaming in",
    "Daniel in the lions den, hands raised in prayer",
    "Mary and the infant Jesus, tender and peaceful",
    "The Last Supper, disciples gathered around the table",
    "A lone cross on a hill at sunset",
    "The Baptism of Jesus in the River Jordan",
    "Angels announcing the birth of Christ to the shepherds",
    "Noah's Ark resting on Mount Ararat after the flood",
    "The Road to Emmaus, two disciples walking with a stranger",
]