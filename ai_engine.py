"""
AI engine — all OpenAI API calls via LangChain live here.

Architecture decisions:
- System prompt assembled dynamically based on denomination + moderation level
- Hallucination guard: model explicitly told to flag uncertain verse refs
- Two-pass for sensitive topics: generate + self-review
- Conversation history passed in full for memory
- LangChain used throughout (langchain + langchain-openai)

MIGRATION NOTES (Gemini → LangChain + OpenAI gpt-4o-mini):
  1. genai.GenerativeModel  →  ChatOpenAI(model="gpt-4o-mini")
  2. model.start_chat(history) + send_message()
     →  ChatOpenAI.invoke([SystemMessage, *history, HumanMessage])
  3. Gemini roles "user"/"model"  →  OpenAI roles "user"/"assistant"
  4. verify_verse_claim uses ChatOpenAI with JSON output parsing
  5. Streaming via LangChain's .stream() iterator
  6. No separate MODEL / MODEL_PRO split needed — gpt-4o-mini handles all tiers
"""

import json
import os
from typing import Optional

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from scripture import (
    get_denomination_context,
    get_translation_for_denomination,
    extract_verse_refs,
    validate_verse_ref,
)

load_dotenv()

from moderation import SYSTEM_PROMPT_SAFETY_ADDENDUM, RiskLevel

# ── OpenAI / LangChain setup ──────────────────────────────────────────────────

_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise ValueError("OPENAI_API_KEY is NOT set in environment variables")
print("Using OpenAI API key:", "SET" if _api_key else "NOT SET")

MODEL = "gpt-4o-mini"   # fast, cheap, capable — replaces both Gemini tiers


# ── LangChain model factory ───────────────────────────────────────────────────

def _make_llm(temperature: float = 0.7, streaming: bool = False) -> ChatOpenAI:
    """Return a configured ChatOpenAI instance."""
    return ChatOpenAI(
        model=MODEL,
        temperature=temperature,
        streaming=streaming,
        openai_api_key=_api_key,
    )


# ── System prompt builder ─────────────────────────────────────────────────────

def build_system_prompt(denomination: str = "general", caution: bool = False) -> str:
    translation  = get_translation_for_denomination(denomination)
    denom_context = get_denomination_context(denomination)

    base = f"""You are Faith & Scripture AI — a knowledgeable, warm, and theologically grounded Christian assistant.

DENOMINATION CONTEXT ({denomination.title()}):
{denom_context}
Preferred Bible translation for this session: {translation}

YOUR ROLE:
- Answer questions about Christianity, theology, church history, and biblical interpretation
- Generate Christian content (prayers, devotionals, reflections, hymn lyrics, sermon outlines)
- Cite scripture accurately — book, chapter, verse
- Maintain a pastoral, respectful, non-judgmental tone
- Acknowledge denominational differences honestly and charitably

HALLUCINATION PREVENTION (HIGHEST PRIORITY):
- Only cite Bible verses you are highly confident exist
- When citing, use format: "Book Chapter:Verse (Translation)"  e.g. "John 3:16 (NIV)"
- If a user quotes a verse that seems wrong or fabricated, gently flag it:
  "I want to make sure we're working with accurate scripture. I can't verify that exact reference.
   The verse you may be thinking of is [correct verse], or please check BibleGateway.com."
- Distinguish clearly: DIRECT QUOTE vs PARAPHRASE vs THEOLOGICAL INTERPRETATION
- Never invent chapter/verse numbers to sound authoritative

TONE & APPROACH:
- Warm, accessible, pastoral — like a knowledgeable pastor/spiritual director
- Intellectually honest: say "Christians disagree on this" when they do
- Graceful with hard questions (theodicy, violence in OT, hell, LGBTQ+, etc.)
- Never preachy or condescending
- For non-Christian users showing interest: welcoming, not pressuring

{SYSTEM_PROMPT_SAFETY_ADDENDUM}
"""

    if caution:
        base += """
⚠️  ELEVATED CAUTION MODE: This conversation has touched on sensitive territory.
Be especially careful, pastoral, and balanced. Do not take extreme positions.
If the user seems distressed, acknowledge their feelings before theology.
"""
    return base


# ── Conversation history converter ────────────────────────────────────────────

def _convert_messages_to_langchain(
    messages: list[dict],
    system_prompt: str,
) -> list:
    """
    Convert OpenAI-style message dicts into LangChain message objects,
    prepending the system prompt.

    Input roles:  "user" | "assistant"
    Output types: SystemMessage | HumanMessage | AIMessage
    """
    lc_messages: list = [SystemMessage(content=system_prompt)]

    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        # ignore any other roles silently

    return lc_messages


# ── Core chat ─────────────────────────────────────────────────────────────────

def chat(
    messages: list[dict],
    denomination: str = "general",
    caution: bool = False,
    stream: bool = False,
) -> str:
    """
    Send conversation to gpt-4o-mini via LangChain and return response text.

    Args:
        messages:     list of {"role": "user"/"assistant", "content": str}
        denomination: e.g. "catholic", "baptist", "orthodox", "general"
        caution:      True → elevated-caution system prompt addendum
        stream:       True → stream tokens (returns joined string)
    """
    system   = build_system_prompt(denomination, caution)
    lc_msgs  = _convert_messages_to_langchain(messages, system)

    if stream:
        llm    = _make_llm(streaming=True)
        chunks = []
        for chunk in llm.stream(lc_msgs):
            # chunk.content is a str (LangChain guarantees this for ChatOpenAI)
            chunks.append(chunk.content or "")
        return "".join(chunks)
    else:
        llm      = _make_llm()
        response = llm.invoke(lc_msgs)
        return response.content


# ── Content generation ────────────────────────────────────────────────────────

def generate_christian_content(
    content_type: str,
    topic: str,
    denomination: str = "general",
    tone: str = "reflective",
) -> str:
    """
    Dedicated content generation: prayers, devotionals, sermon outlines, etc.

    Args:
        content_type: "prayer" | "devotional" | "sermon_outline" | "reflection" | "hymn"
        topic:        Subject matter, e.g. "forgiveness", "Advent hope"
        denomination: Denominational context
        tone:         "reflective" | "celebratory" | "solemn" | "intimate" | "bold"
    """
    type_instructions: dict[str, str] = {
        "prayer": (
            f"Write a heartfelt, scripturally grounded prayer about: {topic}. "
            f"Include at least one relevant scripture reference. Tone: {tone}."
        ),
        "devotional": (
            f"Write a 200-250 word daily devotional on: {topic}. "
            f"Structure: opening verse → reflection → application → closing prayer. "
            f"Verify all scripture citations."
        ),
        "sermon_outline": (
            f"Create a structured sermon outline on: {topic}. "
            f"Include: Title, Key Text, 3 main points each with supporting verses, "
            f"illustration suggestion, and call to action."
        ),
        "reflection": (
            f"Write a thoughtful spiritual reflection on: {topic}. "
            f"Personal, warm, scripturally grounded."
        ),
        "hymn": (
            f"Write original hymn lyrics (2-3 verses + chorus) on the theme of: {topic}. "
            f"Traditional meter preferred."
        ),
    }

    instruction = type_instructions.get(
        content_type,
        f"Create Christian {content_type} content about: {topic}. Tone: {tone}."
    )

    system = build_system_prompt(denomination)
    llm    = _make_llm()
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=instruction),
    ])
    return response.content


# ── Verse verification ────────────────────────────────────────────────────────

_VERSE_VERIFIER_SYSTEM = (
    "You are a precise Biblical reference checker. "
    "You respond ONLY with valid raw JSON — no markdown fences, no preamble, no explanation. "
    "Your sole job is to verify Bible verse references and texts."
)


def _strip_json_fences(text: str) -> str:
    """Remove ```json / ``` wrappers that the model sometimes adds."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def verify_verse_claim(claimed_verse: str, claimed_text: str) -> dict:
    """
    Ask gpt-4o-mini to verify whether a verse reference + text is accurate.

    Returns:
        {
          "reference_exists": bool | None,
          "text_accurate":    bool | None,
          "actual_text":      str,
          "correct_reference": str,
          "notes":            str,
        }
    """
    prompt = f"""A user has cited this Bible verse:

Reference: {claimed_verse}
Claimed text: "{claimed_text}"

Please verify:
1. Does this reference ({claimed_verse}) actually exist in the Bible?
2. Is the quoted text accurate (even approximately)?
3. If inaccurate, what IS the actual text of that verse?
4. If the reference doesn't exist, what verse might they be thinking of?

Respond in this exact JSON format (raw JSON only, no markdown fences):
{{
  "reference_exists": true,
  "text_accurate": true,
  "actual_text": "...",
  "correct_reference": "...",
  "notes": "..."
}}"""

    # Low temperature for factual verification
    llm = _make_llm(temperature=0.0)
    response = llm.invoke([
        SystemMessage(content=_VERSE_VERIFIER_SYSTEM),
        HumanMessage(content=prompt),
    ])

    try:
        cleaned = _strip_json_fences(response.content)
        return json.loads(cleaned)
    except Exception:
        return {
            "reference_exists": None,
            "text_accurate": None,
            "actual_text": "",
            "correct_reference": "",
            "notes": "Could not verify — please check BibleGateway.com",
        }


# ── Image prompt engineering ──────────────────────────────────────────────────

_IMAGE_PROMPT_SYSTEM = """You are an art director specialising in reverent Christian imagery.
Convert user requests into detailed, respectful image generation prompts.
Style should be: painterly, classical, warm lighting, inspired by Renaissance/Byzantine art traditions.
Never include: violence, sexuality, mockery, or theologically controversial depictions.
If the request is for something potentially inappropriate, redirect to beautiful symbolic imagery instead.
Respond with ONLY the image prompt, nothing else — no preamble, no explanation."""


def generate_image_prompt(user_request: str, denomination: str = "general") -> str:
    """
    Convert a user's image request into a safe, detailed, reverent image generation prompt.
    Acts as a safety + quality buffer between raw user input and the image API.

    Returns:
        A polished image-generation prompt string (no extra text).
    """
    llm = _make_llm(temperature=0.5)
    response = llm.invoke([
        SystemMessage(content=_IMAGE_PROMPT_SYSTEM),
        HumanMessage(content=f"Create an image prompt for: {user_request}"),
    ])
    return response.content.strip()


# ── Difficult theology handler ────────────────────────────────────────────────

def handle_difficult_theology(question: str, denomination: str = "general") -> str:
    """
    Specially tuned handler for theodicy, evil, suffering,
    religious violence, and other hard theological questions.

    Uses two-pass approach:
      Pass 1 — Generate a thorough theological response
      Pass 2 — Self-review for balance, pastoral care, and accuracy
    """
    extra = """

SPECIAL MODE: Difficult Theological Question
- Acknowledge the genuine difficulty and emotional weight of the question
- Present the major theological positions (e.g. Augustine, Plantinga on theodicy)
- Do not pretend there are easy answers
- Cite relevant scripture with context, not proof-texting
- End with pastoral care, not a triumphalist conclusion
- Be honest about what Christianity has historically said AND its limits
"""
    system = build_system_prompt(denomination) + extra
    llm    = _make_llm(temperature=0.5)

    # Pass 1: Generate
    draft_response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=question),
    ])
    draft = draft_response.content

    # Pass 2: Self-review for balance and pastoral care
    review_prompt = f"""Review the following theological response for:
1. Theological balance — does it represent multiple legitimate Christian perspectives?
2. Pastoral sensitivity — is it appropriately compassionate?
3. Scripture accuracy — are all citations verifiable?
4. Tone — is it honest without being dismissive or triumphalist?

If the response is good, return it as-is.
If it needs improvement, return an improved version.
Return ONLY the final response text, no commentary.

Response to review:
---
{draft}
---"""

    final_response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=review_prompt),
    ])
    return final_response.content