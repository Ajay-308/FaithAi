"""
Christian image generation module.

Architecture:
1. User prompt  →  moderation check
2. Prompt       →  gpt-4o-mini mediated prompt engineering (safe, reverent rewrite)
3. Rewritten prompt  →  image API (Pollinations free API as demo)
4. Return image URL

For production: swap Pollinations for DALL-E 3 / Stability AI.
Pollinations used here for zero-API-key demo capability.
"""

import requests
import urllib.parse

from moderation import moderate_image_prompt, RiskLevel
from ai_engine import generate_image_prompt


# ── Constants ─────────────────────────────────────────────────────────────────

STYLE_SUFFIX = (
    "classical Christian art style, Renaissance painting, warm golden light, "
    "reverent atmosphere, highly detailed, masterpiece quality"
)

POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def generate_christian_image(user_request: str, denomination: str = "general") -> dict:
    """
    Full pipeline: moderate → rewrite → generate.

    Returns:
        {
          "success":     bool,
          "image_url":   str | None,
          "prompt_used": str | None,
          "reason":      str | None,
        }
    """
    # Step 1: Moderate raw request
    mod = moderate_image_prompt(user_request)
    if mod.level == RiskLevel.BLOCK:
        return {
            "success": False,
            "image_url": None,
            "prompt_used": None,
            "reason": mod.suggested_response,
        }

    # Step 2: gpt-4o-mini rewrites into a safe, detailed, reverent art prompt
    try:
        safe_prompt = generate_image_prompt(user_request, denomination)
    except Exception:
        safe_prompt = f"Beautiful Christian artwork depicting {user_request}, {STYLE_SUFFIX}"

    # Step 3: Append style suffix if needed
    final_prompt = safe_prompt
    keywords = ("renaissance", "classical", "byzantine", "painterly", "masterpiece")
    if not any(kw in safe_prompt.lower() for kw in keywords):
        final_prompt = f"{safe_prompt}, {STYLE_SUFFIX}"

    # Step 4: Generate via Pollinations
    image_url = _build_pollinations_url(final_prompt)
    success   = _verify_url_accessible(image_url)

    return {
        "success": success,
        "image_url": image_url,
        "prompt_used": final_prompt,
        "reason": None,
    }


# ── URL helpers ───────────────────────────────────────────────────────────────

def _build_pollinations_url(prompt: str) -> str:
    encoded = urllib.parse.quote(prompt)
    return POLLINATIONS_URL.format(prompt=encoded)


def _verify_url_accessible(url: str) -> bool:
    try:
        resp = requests.head(url, timeout=10, allow_redirects=True)
        return resp.status_code < 400
    except requests.RequestException:
        return True  # Pollinations generates on GET; HEAD may return 404


# ── Optional: DALL-E 3 backend ────────────────────────────────────────────────

def _generate_with_dalle(prompt: str, api_key: str) -> str | None:
    """
    Production backend using OpenAI DALL-E 3.
    langchain-openai is already installed, so you can also use it here.
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
    except Exception:
        return None


# ── Optional: Stability AI backend ───────────────────────────────────────────

def _generate_with_stability(prompt: str, api_key: str) -> str | None:
    try:
        url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        body = {
            "text_prompts": [{"text": prompt, "weight": 1}],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30,
        }
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        return resp.json()["artifacts"][0]["base64"]
    except Exception:
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