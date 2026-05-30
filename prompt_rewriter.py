"""
Prompt Rewriter — isolated module to break circular import.

  ai_engine.py  →  image_gen.py  →  ai_engine.py   ← WAS circular
  ai_engine.py  →  prompt_rewriter.py               ← FIXED
  image_gen.py  →  prompt_rewriter.py               ← FIXED
"""

from __future__ import annotations
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


_llm_precise = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


def generate_image_prompt(user_request: str, denomination: str = "general") -> str:
    """
    Rewrite a user image request into a safe, detailed, art-directed prompt
    suitable for Stability AI / Pollinations.
    """
    system = """You are a Christian art director specializing in reverent religious imagery.
Rewrite the user's request into a detailed, safe image generation prompt.
Style: Renaissance painting, classical Christian art, warm golden light, reverent atmosphere.
Never include: violence, disturbing imagery, disrespectful depictions of sacred figures.
Return ONLY the rewritten prompt, no explanation."""

    response = _llm_precise.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Create an image prompt for: {user_request}"),
    ])
    return response.content.strip()