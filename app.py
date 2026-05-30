"""
Faith & Scripture AI — Streamlit App
=====================================
A Christianity-focused AI assistant with:
- Scripture-grounded chat with hallucination prevention
- Christian content generation (prayers, devotionals, sermons)
- Christian image generation with safety layer
- Denomination-aware handling
- Moderation / safety throughout
- Eval dashboard

MIGRATION: Uses LangChain + OpenAI gpt-4o-mini (was Gemini).
Set OPENAI_API_KEY in your .env file.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import json
from moderation import moderate_message, moderate_image_prompt, RiskLevel
from ai_engine import chat, generate_christian_content, verify_verse_claim, handle_difficult_theology
from scripture import extract_verse_refs, validate_verse_ref, get_denomination_context
from image_gen import generate_christian_image, EXAMPLE_IMAGE_PROMPTS
from data.dataset import EVAL_DATASET, run_moderation_eval

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Faith & Scripture AI",
    page_icon="✝️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2d6a8c 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p { color: #b8d4e8; margin: 0.3rem 0 0; font-size: 0.95rem; }

    .verse-box {
        background: #f0f7ff;
        border-left: 4px solid #1a3a5c;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-style: italic;
        color: #1a3a5c;
    }
    .safety-badge-safe {
        background: #d4edda; color: #155724;
        padding: 0.2rem 0.6rem; border-radius: 12px;
        font-size: 0.8rem; font-weight: 600;
    }
    .safety-badge-block {
        background: #f8d7da; color: #721c24;
        padding: 0.2rem 0.6rem; border-radius: 12px;
        font-size: 0.8rem; font-weight: 600;
    }
    .safety-badge-caution {
        background: #fff3cd; color: #856404;
        padding: 0.2rem 0.6rem; border-radius: 12px;
        font-size: 0.8rem; font-weight: 600;
    }
    .denom-info {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #444;
        margin-top: 0.5rem;
    }
    .eval-pass { color: #155724; font-weight: bold; }
    .eval-fail { color: #721c24; font-weight: bold; }
    .stChatMessage { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

defaults = {
    "messages": [],
    "denomination": "general",
    "caution_mode": False,
    "moderation_log": [],
    "image_prompt": "",
    "tool_mode": "💬 Chat",
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ✝️ Faith & Scripture AI")
    st.caption("Grounded in Scripture · Respectful of Tradition")

    st.divider()

    st.markdown("### 🕊️ Denomination")
    denomination = st.selectbox(
        "Select your tradition",
        ["general", "protestant", "catholic", "orthodox", "reformed", "baptist", "lutheran"],
        format_func=lambda x: x.title(),
        key="denomination",
    )

    if denomination != "general":
        denom_info = get_denomination_context(denomination) or ""
        if denom_info:
            st.markdown(f'<div class="denom-info">{denom_info[:200]}...</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📖 Quick Tools")
    tool_mode = st.radio(
        "Mode",
        ["💬 Chat", "✍️ Content Generator", "🖼️ Image Generator", "🔍 Verse Verifier", "🧪 Eval Dashboard"],
        label_visibility="collapsed",
        key="tool_mode",
    )

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.caution_mode = False
        st.rerun()

    st.divider()

    with st.expander("🛡️ Moderation Log", expanded=False):
        if st.session_state.moderation_log:
            for entry in st.session_state.moderation_log[-5:]:
                level = entry["level"]
                color = {"safe": "🟢", "caution": "🟡", "block": "🔴"}.get(level, "⚪")
                st.caption(f"{color} {entry['preview'][:40]}...")
                if entry.get("reason"):
                    st.caption(f"   ↳ {entry['reason']}")
        else:
            st.caption("No moderation events yet.")

    st.divider()
    # Updated: reflects the new model/provider
    st.caption("Built with LangChain · OpenAI gpt-4o-mini")
    st.caption("Architecture: Prompt Engineering + Grounding + Safety Layers")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>✝️ Faith & Scripture AI</h1>
    <p>A theologically grounded assistant · Scripture-aware · Denomination-sensitive · Safety-first</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: log moderation event
# ─────────────────────────────────────────────────────────────────────────────

def log_moderation(user_input: str, level: str, reason: str = ""):
    st.session_state.moderation_log.append({
        "preview": user_input[:60],
        "level": level,
        "reason": reason,
    })


# ═════════════════════════════════════════════════════════════════════════════
# MODE: CHAT
# ═════════════════════════════════════════════════════════════════════════════

if st.session_state.tool_mode == "💬 Chat":

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🙏" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    if not st.session_state.messages:
        st.markdown("#### 💡 Try asking...")
        cols = st.columns(2)
        suggestions = [
            "What does John 3:16 really mean?",
            "Explain the Trinity in simple terms",
            "What does the Bible say about anxiety?",
            "Why did God allow suffering in Job's life?",
            "Hezekiah 4:11 says trust in the Lord — explain this",  # hallucination trap
        ]
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": s})
                st.rerun()

    user_input = st.chat_input("Ask anything about Christianity, scripture, theology...")

    if user_input:
        mod = moderate_message(user_input)
        log_moderation(user_input, mod.level.value, mod.reason or "")

        if mod.level == RiskLevel.BLOCK:
            st.session_state.messages.append({"role": "user", "content": user_input})
            response = mod.suggested_response
            st.session_state.messages.append({"role": "assistant", "content": f"🛡️ **Content Policy** | {response}"})
            st.rerun()

        elif mod.level == RiskLevel.CAUTION:
            st.session_state.caution_mode = True

        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        difficult_keywords = ["suffering", "evil", "genocide", "homosexual", "hell", "theodicy", "why did god"]
        is_difficult = any(kw in user_input.lower() for kw in difficult_keywords)

        with st.chat_message("assistant", avatar="🙏"):
            with st.spinner("Searching scripture..."):
                try:
                    if is_difficult:
                        response = handle_difficult_theology(
                            user_input,
                            denomination=st.session_state.denomination,
                        )
                    else:
                        response = chat(
                            messages=st.session_state.messages,
                            denomination=st.session_state.denomination,
                            caution=st.session_state.caution_mode,
                        )
                except Exception as e:
                    response = (
                        f"⚠️ I encountered an error: {str(e)}\n\n"
                        "Please ensure your OPENAI_API_KEY is set in your .env file."
                    )

            if st.session_state.caution_mode:
                st.caption("🟡 Sensitive topic — responding with extra pastoral care")

            st.markdown(response)

            refs = extract_verse_refs(response)
            if refs:
                with st.expander(f"📖 Scripture references in this response ({len(refs)})", expanded=False):
                    for ref in refs:
                        st.markdown(
                            f"• **{ref}** — "
                            f"[Look up on BibleGateway](https://www.biblegateway.com/passage/?search={ref.replace(' ', '+')})"
                        )

        st.session_state.messages.append({"role": "assistant", "content": response})


# ═════════════════════════════════════════════════════════════════════════════
# MODE: CONTENT GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.tool_mode == "✍️ Content Generator":

    st.markdown("### ✍️ Christian Content Generator")
    st.caption("Generate prayers, devotionals, sermon outlines, hymns — all scripture-grounded")

    col1, col2 = st.columns([1, 1])

    with col1:
        content_type = st.selectbox(
            "Content type",
            ["prayer", "devotional", "sermon_outline", "reflection", "hymn"],
            format_func=lambda x: x.replace("_", " ").title(),
        )
        topic = st.text_input(
            "Topic or theme",
            placeholder="e.g. Forgiveness, Hope in suffering, God's grace...",
        )
        tone = st.selectbox(
            "Tone",
            ["reflective", "celebratory", "penitential", "hopeful", "instructive"],
        )
        generate_btn = st.button("✨ Generate", type="primary", use_container_width=True)

    with col2:
        if generate_btn and topic:
            mod = moderate_message(topic)
            if mod.level == RiskLevel.BLOCK:
                st.error(f"🛡️ {mod.suggested_response}")
            else:
                with st.spinner(f"Crafting your {content_type}..."):
                    try:
                        result = generate_christian_content(
                            content_type=content_type,
                            topic=topic,
                            denomination=st.session_state.denomination,
                            tone=tone,
                        )
                        st.markdown(f'<div class="verse-box">{result}</div>', unsafe_allow_html=True)
                        refs = extract_verse_refs(result)
                        if refs:
                            st.caption(f"📖 Scripture cited: {', '.join(refs)}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        elif not topic and generate_btn:
            st.warning("Please enter a topic first.")
        else:
            st.info("👈 Fill in the form and click Generate")


# ═════════════════════════════════════════════════════════════════════════════
# MODE: IMAGE GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.tool_mode == "🖼️ Image Generator":

    st.markdown("### 🖼️ Christian Image Generator")
    st.caption("Generate reverent Christian artwork · All prompts safety-checked and art-directed")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.session_state.image_prompt = st.text_area(
            "Describe the image",
            value=st.session_state.image_prompt,
            placeholder="e.g. Jesus walking on water at sunset...",
            height=100,
            key="image_prompt_area",
        )

        st.markdown("**Or choose an example:**")
        for ex in EXAMPLE_IMAGE_PROMPTS[:4]:
            if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
                st.session_state.image_prompt = ex
                st.rerun()

        gen_btn = st.button("🎨 Generate Image", type="primary", use_container_width=True)

    with col2:
        current_prompt = st.session_state.image_prompt

        if gen_btn and current_prompt:
            mod = moderate_image_prompt(current_prompt)
            log_moderation(current_prompt, mod.level.value, mod.reason or "")

            if mod.level == RiskLevel.BLOCK:
                st.error(f"🛡️ {mod.suggested_response}")
            else:
                with st.spinner("Creating your Christian artwork..."):
                    try:
                        result = generate_christian_image(
                            current_prompt,
                            denomination=st.session_state.denomination,
                        )
                        if result["success"]:
                            st.image(result["image_url"], caption="Generated Christian Artwork", use_container_width=True)
                            with st.expander("🎨 Art direction prompt used"):
                                st.caption(result["prompt_used"])
                        else:
                            st.error(result["reason"])
                    except Exception as e:
                        st.error(f"Image generation error: {e}")
        elif gen_btn and not current_prompt:
            st.warning("Please describe the image you want.")
        else:
            st.info("👈 Describe a scene and click Generate")


# ═════════════════════════════════════════════════════════════════════════════
# MODE: VERSE VERIFIER
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.tool_mode == "🔍 Verse Verifier":

    st.markdown("### 🔍 Bible Verse Verifier")
    st.caption("Check if a verse reference and text are accurate — guards against hallucination and misquotation")

    col1, col2 = st.columns([1, 1])

    with col1:
        verse_ref  = st.text_input("Verse reference", placeholder="e.g. John 3:16")
        verse_text = st.text_area("Verse text to verify", placeholder="Paste the verse text here...", height=100)
        verify_btn = st.button("🔍 Verify", type="primary")

        st.divider()
        st.markdown("**Quick structural check:**")
        book_check  = st.text_input("Book name", placeholder="e.g. Hezekiah")
        chap_check  = st.number_input("Chapter", min_value=1, max_value=200, value=1)
        verse_check = st.number_input("Verse",   min_value=1, max_value=200, value=1)
        struct_btn  = st.button("Check structure")

    with col2:
        if verify_btn and verse_ref and verse_text:
            with st.spinner("Verifying against scripture..."):
                try:
                    result = verify_verse_claim(verse_ref, verse_text)
                    if result.get("reference_exists") is True:
                        st.success("✅ Reference exists in the Bible")
                    elif result.get("reference_exists") is False:
                        st.error("❌ This reference does NOT exist in the Bible")
                    else:
                        st.warning("⚠️ Could not verify reference definitively")

                    if result.get("text_accurate") is True:
                        st.success("✅ Text is accurate")
                    elif result.get("text_accurate") is False:
                        st.error("❌ Text does not match this verse")

                    if result.get("actual_text"):
                        st.markdown("**Actual verse text:**")
                        st.markdown(f'<div class="verse-box">{result["actual_text"]}</div>', unsafe_allow_html=True)

                    if result.get("correct_reference") and result["correct_reference"] != verse_ref:
                        st.info(f"📖 Correct reference: **{result['correct_reference']}**")

                    if result.get("notes"):
                        st.caption(f"Note: {result['notes']}")
                except Exception as e:
                    st.error(f"Verification error: {e}")

        if struct_btn and book_check:
            is_valid, msg = validate_verse_ref(book_check, int(chap_check), int(verse_check))
            if is_valid:
                st.success(f"✅ {msg} {chap_check}:{verse_check} — structurally valid reference")
            else:
                st.error(f"❌ {msg}")


# ═════════════════════════════════════════════════════════════════════════════
# MODE: EVAL DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.tool_mode == "🧪 Eval Dashboard":

    st.markdown("### 🧪 Evaluation Dashboard")
    st.caption("Test suite covering hallucination, adversarial prompts, edge cases, and image safety")

    tab1, tab2, tab3 = st.tabs(["📋 Dataset", "🏃 Run Moderation Eval", "📊 Results"])

    with tab1:
        category = st.selectbox(
            "Category",
            list(EVAL_DATASET.keys()),
            format_func=lambda x: x.replace("_", " ").title(),
        )
        for case in EVAL_DATASET[category]:
            with st.expander(f"**{case['id']}** — {case['input'][:70]}..."):
                st.markdown(f"**Input:** {case['input']}")
                st.markdown(f"**Expected:** {case['expected_behaviour']}")
                if "red_flag" in case:
                    st.error(f"🚩 Red flag: {case['red_flag']}")

    with tab2:
        st.markdown("Run the automated moderation layer against known-bad inputs.")
        st.caption("Tests adversarial prompts + image prompts against pattern-based moderation.")

        if st.button("▶️ Run Moderation Eval", type="primary"):
            with st.spinner("Running evaluations..."):
                results = run_moderation_eval()

            total     = results["pass"] + results["fail"]
            pass_rate = (results["pass"] / total * 100) if total > 0 else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Tests", total)
            c2.metric("Passed",      results["pass"])
            c3.metric("Pass Rate",   f"{pass_rate:.0f}%")

            st.progress(pass_rate / 100)
            st.markdown("**Detailed results:**")
            for detail in results["details"]:
                icon = "✅" if detail["passed"] else "❌"
                st.markdown(f"{icon} **{detail['id']}** — `{detail['input_preview']}` → `{detail['result']}`")

    with tab3:
        st.markdown("### Architecture Summary")
        st.markdown("""
**Hallucination Prevention**
- System prompt explicitly forbids fabricating verse references
- `verify_verse_claim()` powered by gpt-4o-mini with temperature=0 for factual precision
- Structural validation (book name lookup) catches impossible references
- Model instructed to distinguish: direct quote / paraphrase / interpretation

**Safety Layers (in order)**
1. **Pre-flight**: Regex pattern matching — instant, free, no API call
2. **Prompt engineering**: System prompt with safety addendum on every LangChain call
3. **Semantic router**: Difficult theology keywords → `handle_difficult_theology()` two-pass
4. **Image safety**: Separate image moderation + gpt-4o-mini prompt rewriting
5. **Post-generation**: Verse reference extraction + BibleGateway links for verification

**Denomination Awareness**
- 7 denominations with distinct theological notes and preferred translations
- Deuterocanonical books context for Catholic/Orthodox
- Caution flag passed to system prompt for sensitive interactions

**Model**
- All LLM calls: `gpt-4o-mini` via LangChain `ChatOpenAI`
- Verse verifier uses `temperature=0` for factual accuracy
- Two-pass difficult theology uses `temperature=0.5` for nuanced but stable output
        """)