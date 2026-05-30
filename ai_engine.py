"""
AI Engine — Faith & Scripture AI
==================================
All LLM calls go through this module.

KEY CHANGE: Every chat/content call now goes through retrieval first.
  User Query → scripture_retrieval → context block → LLM
  The LLM is explicitly instructed to use ONLY the retrieved verses.
  This prevents verse fabrication.

Model: gpt-4o-mini via LangChain ChatOpenAI
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Import retrieval layer
from scripture_retrival import (
    retrieve_scripture_context,
    format_context_block,
    validate_book_name,
    get_verse,
)

# ── Model setup ───────────────────────────────────────────────────────────────

_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

_llm_precise = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# ── System prompts ────────────────────────────────────────────────────────────

_BASE_SYSTEM = """You are a knowledgeable, respectful, and pastoral Christian AI assistant.

CRITICAL RULES — follow without exception:
1. NEVER invent or fabricate Bible verses or references.
2. When scripture context is provided below, answer ONLY using those verses.
3. If no scripture context is provided, say you cannot cite a specific verse but can discuss the theme generally.
4. If asked about a verse not in the provided context, say: "I don't have that verse in my current dataset to quote accurately."
5. Always clearly distinguish between: direct quote / paraphrase / general Christian teaching.
6. Be respectful of all Christian traditions. When traditions differ, explain each view.
7. Refuse to use scripture to promote hatred, discrimination, or harm.
"""

_HALLUCINATION_GUARD = """
HALLUCINATION PREVENTION:
- Do NOT quote verses from memory. Only quote what is in the SCRIPTURE CONTEXT above.
- If the user mentions a verse not in the context, acknowledge it exists (if it does) but say you cannot quote it accurately without the text.
- Never complete a partial verse from memory.
"""


# ── Main chat function ────────────────────────────────────────────────────────

def chat(
    messages: list[dict],
    denomination: str = "general",
    caution: bool = False,
) -> str:
    """
    Retrieval-first chat.
    1. Extract the latest user message
    2. Retrieve relevant scripture from local dataset
    3. Build grounded prompt
    4. Call LLM
    """
    if not messages:
        return "Please ask me a question about Christianity or scripture."

    latest_user_msg = ""
    for m in reversed(messages):
        if m["role"] == "user":
            latest_user_msg = m["content"]
            break

    # ── Step 1: Check for fake book references ────────────────────────────────
    fake_ref = _detect_fake_reference(latest_user_msg)
    if fake_ref:
        return fake_ref

    # ── Step 2: Retrieve scripture context ────────────────────────────────────
    retrieved = retrieve_scripture_context(latest_user_msg, top_k=4)
    context_block = format_context_block(retrieved) if retrieved else ""

    # ── Step 3: Build system prompt ───────────────────────────────────────────
    denom_note = _get_denomination_note(denomination)
    caution_note = "\nThis is a sensitive topic. Respond with extra pastoral care and compassion." if caution else ""

    system_content = _BASE_SYSTEM
    if context_block:
        system_content += f"\n\n{context_block}\n{_HALLUCINATION_GUARD}"
    else:
        system_content += "\n\nNo specific scripture has been retrieved for this query. Discuss the theme thoughtfully but do not fabricate verse references."
    system_content += denom_note + caution_note

    # ── Step 4: Build message history ─────────────────────────────────────────
    lc_messages = [SystemMessage(content=system_content)]
    for m in messages[:-1]:   # all but last (latest already captured)
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))
    lc_messages.append(HumanMessage(content=latest_user_msg))

    response = _llm.invoke(lc_messages)
    return response.content


# ── Difficult theology ────────────────────────────────────────────────────────

def handle_difficult_theology(question: str, denomination: str = "general") -> str:
    """
    Two-pass approach for theologically sensitive questions.
    Pass 1: retrieve relevant verses
    Pass 2: generate nuanced, multi-denominational response grounded in those verses
    """
    # Retrieve relevant scripture
    retrieved = retrieve_scripture_context(question, top_k=5)
    context_block = format_context_block(retrieved) if retrieved else ""

    system = f"""{_BASE_SYSTEM}

{context_block}

{_HALLUCINATION_GUARD}

This is a theologically difficult question. Your response must:
1. Acknowledge the difficulty honestly
2. Present multiple Christian perspectives (Catholic, Protestant, Orthodox where relevant)
3. Ground reasoning in the scripture provided above
4. Avoid declaring one tradition absolutely correct on disputed matters
5. Be pastorally sensitive — someone may be hurting
6. End with pastoral encouragement

{_get_denomination_note(denomination)}
"""

    response = _llm_precise.invoke([
        SystemMessage(content=system),
        HumanMessage(content=question),
    ])
    return response.content


# ── Content generation ────────────────────────────────────────────────────────

def generate_christian_content(
    content_type: str,
    topic: str,
    denomination: str = "general",
    tone: str = "reflective",
) -> str:
    """
    Generate prayers, devotionals, sermon outlines, etc.
    Retrieves relevant scripture first so content is grounded.
    """
    retrieved = retrieve_scripture_context(topic, top_k=4)
    context_block = format_context_block(retrieved) if retrieved else ""

    type_instructions = {
        "prayer":          "Write a heartfelt Christian prayer of 150-200 words.",
        "devotional":      "Write a daily devotional of 200-250 words with a scripture focus, reflection, and application.",
        "sermon_outline":  "Create a structured sermon outline with: Title, Main Scripture, 3 points each with sub-points and supporting verses, and a conclusion.",
        "reflection":      "Write a thoughtful spiritual reflection of 200 words.",
        "hymn":            "Write original hymn lyrics with 3 verses and a chorus in a traditional style.",
    }

    instruction = type_instructions.get(content_type, f"Write Christian {content_type} content.")

    system = f"""{_BASE_SYSTEM}

{context_block}

{_HALLUCINATION_GUARD}

Task: {instruction}
Topic: {topic}
Tone: {tone}
Denomination context: {denomination}

Only cite scripture that appears in the SCRIPTURE CONTEXT above.
If you reference a verse, quote it accurately from the context provided.
"""

    response = _llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Generate {content_type} about: {topic}"),
    ])
    return response.content


# ── Verse verification ────────────────────────────────────────────────────────

def verify_verse_claim(reference: str, claimed_text: str) -> dict:
    """
    Verify whether a verse reference and text are accurate.
    First checks local dataset, then uses LLM with temperature=0.
    """
    # ── Step 1: Check local dataset first ────────────────────────────────────
    local_verse = get_verse(reference)
    if local_verse:
        actual_text   = local_verse["text"]
        text_accurate = _texts_similar(claimed_text, actual_text)
        return {
            "reference_exists": True,
            "text_accurate":    text_accurate,
            "actual_text":      actual_text,
            "correct_reference": reference,
            "notes":            "Verified from local Bible dataset (NIV).",
        }

    # ── Step 2: Check if book exists at all ───────────────────────────────────
    parts     = reference.strip().rsplit(" ", 1)
    book_name = parts[0] if len(parts) == 2 else reference
    valid_book, book_msg = validate_book_name(book_name)
    if not valid_book:
        return {
            "reference_exists": False,
            "text_accurate":    False,
            "actual_text":      None,
            "correct_reference": None,
            "notes":            book_msg,
        }

    # ── Step 3: Book exists but verse not in local dataset → ask LLM ─────────
    system = """You are a Bible fact-checker. Be precise and honest.
If you are not certain, say so. Do NOT invent verse text.
Respond in JSON with keys:
  reference_exists (bool),
  text_accurate (bool or null if uncertain),
  actual_text (string or null),
  correct_reference (string or null),
  notes (string)
"""
    prompt = (
        f"Reference claimed: {reference}\n"
        f"Text claimed: {claimed_text}\n\n"
        "Does this reference exist? Is the text accurate? "
        "This verse is not in my local dataset so answer carefully."
    )

    import json as _json
    try:
        response = _llm_precise.invoke([
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return _json.loads(raw)
    except Exception:
        return {
            "reference_exists": None,
            "text_accurate":    None,
            "actual_text":      None,
            "correct_reference": None,
            "notes":            "Could not verify — not in local dataset. Check a Bible concordance.",
        }


# ── Image prompt rewriter ─────────────────────────────────────────────────────

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


# ── Internal helpers ──────────────────────────────────────────────────────────

def _detect_fake_reference(query: str) -> str | None:
    """
    Check if query references a non-existent Bible book.
    Returns an explanatory message if fake, None if query seems fine.
    """
    import re
    pattern = r'\b([1-3]?\s?[A-Za-z]+)\s+\d+:\d+\b'
    matches = re.findall(pattern, query)

    for raw_book in matches:
        book = raw_book.strip().title()
        valid, msg = validate_book_name(book)
        if not valid:
            return (
                f"⚠️ **Reference Issue**: {msg}\n\n"
                "I won't fabricate content for a verse that doesn't exist. "
                "Please check the reference and try again. "
                "You can use the **Verse Verifier** tool to validate references."
            )
    return None


def _get_denomination_note(denomination: str) -> str:
    notes = {
        "catholic":   "\nDenomination: Catholic. Include references to Tradition and Magisterium where relevant. Deuterocanonical books are canonical.",
        "orthodox":   "\nDenomination: Eastern Orthodox. Emphasize Theosis, Holy Tradition, and the Church Fathers. Deuterocanonical books are canonical.",
        "protestant": "\nDenomination: Protestant. Emphasize Sola Scriptura. Scripture alone is the final authority.",
        "reformed":   "\nDenomination: Reformed/Calvinist. Emphasize God's sovereignty, election, and covenant theology.",
        "baptist":    "\nDenomination: Baptist. Emphasize believer's baptism, local church authority, and scripture alone.",
        "lutheran":   "\nDenomination: Lutheran. Emphasize Law and Gospel distinction, grace alone, faith alone.",
        "general":    "",
    }
    return notes.get(denomination, "")


def _texts_similar(text_a: str, text_b: str) -> bool:
    """
    Simple similarity check: do the texts share most key words?
    """
    import re
    stop = {"the", "a", "an", "and", "or", "is", "are", "was", "were", "in", "of", "to", "for"}

    def words(t):
        return set(re.findall(r"[a-z]+", t.lower())) - stop

    w_a = words(text_a)
    w_b = words(text_b)
    if not w_a or not w_b:
        return False
    overlap = len(w_a & w_b) / max(len(w_a), len(w_b))
    return overlap > 0.6