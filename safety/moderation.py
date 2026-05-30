"""
Moderation layer — risk classification for user inputs.

Two surfaces:
  1. moderate_message()       — for chat messages
  2. moderate_image_prompt()  — for image generation requests

RiskLevel enum drives routing:
  SAFE    → proceed normally
  CAUTION → proceed with caution flag sent to ai_engine
  BLOCK   → return refusal, never reach LLM
"""

from dataclasses import dataclass
from enum import Enum
import re


#  Risk levels ─

class RiskLevel(Enum):
    SAFE    = "safe"
    CAUTION = "caution"
    BLOCK   = "block"


@dataclass
class ModerationResult:
    level:             RiskLevel
    reason:            str        # internal logging label
    suggested_response: str       # user-facing message (only shown on BLOCK)


#  Keyword lists ─

_BLOCK_PATTERNS = [
    r"\b(make|build|create|generate)\b.{0,30}\b(bomb|weapon|explosive|poison|virus|malware)\b",
    r"\b(self.?harm|suicide|kill\s+my?self)\b",
    r"\bchild.{0,10}(sex|nude|naked|explicit)\b",
    r"\b(porn|pornograph|hentai|explicit\s+sex)\b",
    r"\b(hate\s+speech|racial\s+slur)\b",
    r"\b(satanic\s+ritual\s+abuse|desecrat)\b",
]

_CAUTION_PATTERNS = [
    r"\b(suicide|self.?harm|depression|hopeless|end\s+it)\b",
    r"\b(abuse|trauma|assault)\b",
    r"\b(hell|damnation|going\s+to\s+hell)\b",
    r"\b(lgbtq|homosexual|gay|transgender)\b",
    r"\b(abortion|pro.?life|pro.?choice)\b",
    r"\b(israel|palestine|jihad|crusade)\b",
    r"\b(cult|false\s+prophet|heresy|heretic)\b",
    r"\b(doubt|deconstructing|losing\s+faith|leaving\s+the\s+church)\b",
]

_IMAGE_BLOCK_PATTERNS = [
    r"\b(nude|naked|explicit|sexual|erotic|porn)\b",
    r"\b(gore|blood|violence|war|battle|weapon)\b",
    r"\b(satan|devil|demon|666|antichrist)\b",
    r"\b(mock|ridicul|blasphemy|blasphemous)\b",
    r"\b(real\s+person|photo.?realistic|deepfake)\b",
]


#  Moderation functions 

def _matches_any(text: str, patterns: list[str]) -> str | None:
    """Return the first matching pattern label, or None."""
    lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, lower):
            return pattern
    return None


def moderate_message(text: str) -> ModerationResult:
    if not text or not text.strip():
        return ModerationResult(
            level=RiskLevel.SAFE,
            reason="empty_input",
            suggested_response="",
        )

    matched_block = _matches_any(text, _BLOCK_PATTERNS)
    if matched_block:
        return ModerationResult(
            level=RiskLevel.BLOCK,
            reason=f"block_pattern: {matched_block}",
            suggested_response=(
                "I'm not able to help with that request. "
                "If you're in distress, please reach out to a trusted person or a crisis helpline."
            ),
        )

    matched_caution = _matches_any(text, _CAUTION_PATTERNS)
    if matched_caution:
        return ModerationResult(
            level=RiskLevel.CAUTION,
            reason=f"caution_pattern: {matched_caution}",
            suggested_response="",
        )

    return ModerationResult(
        level=RiskLevel.SAFE,
        reason="clean",
        suggested_response="",
    )


def moderate_image_prompt(text: str) -> ModerationResult:
    if not text or not text.strip():
        return ModerationResult(
            level=RiskLevel.BLOCK,
            reason="empty_image_prompt",
            suggested_response="Please describe what image you'd like to create.",
        )

    lower = text.lower()

    matched = _matches_any(lower, _IMAGE_BLOCK_PATTERNS)
    if matched:
        return ModerationResult(
            level=RiskLevel.BLOCK,
            reason=f"image_block_pattern: {matched}",
            suggested_response=(
                "I can only generate reverent, uplifting Christian imagery. "
                "That request falls outside what I'm able to create. "
                "Try something like: 'The Good Shepherd at sunset' or 'A peaceful church interior'."
            ),
        )

    matched_text = _matches_any(lower, _BLOCK_PATTERNS)
    if matched_text:
        return ModerationResult(
            level=RiskLevel.BLOCK,
            reason=f"standard_block_in_image: {matched_text}",
            suggested_response=(
                "I'm not able to generate that image. "
                "Please request peaceful, reverent Christian artwork."
            ),
        )

    return ModerationResult(
        level=RiskLevel.SAFE,
        reason="clean_image_prompt",
        suggested_response="",
    )


#  System prompt safety addendum ─

SYSTEM_PROMPT_SAFETY_ADDENDUM = """
SAFETY GUIDELINES:
- Never provide information that could facilitate self-harm, violence, or illegal activity
- If a user appears to be in crisis (suicidal, abusive situation), respond with compassion
  and gently suggest professional resources (e.g. a pastor, counselor, or crisis line)
- Do not produce content that demeans any religious group, ethnicity, gender, or sexuality
- For politically charged topics (abortion, LGBTQ+, politics), present multiple Christian
  perspectives charitably without advocating for one political party
- Refuse requests to impersonate specific real-world religious leaders in misleading ways
- Do not generate content that could be used as cult recruitment or spiritually manipulative material
"""