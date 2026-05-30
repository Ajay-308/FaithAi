"""
Faith & Scripture AI — Streamlit App
=====================================
A Christianity-focused AI assistant with:
- Scripture-grounded chat (retrieval-first, hallucination-prevented)
- Christian content generation (prayers, devotionals, sermons)
- Christian image generation with safety layer
- Denomination-aware handling
- Moderation / safety throughout
- Eval dashboard

Set OPENAI_API_KEY and STABILITY_API_KEY in your .env file.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from safety.moderation import moderate_message, moderate_image_prompt, RiskLevel
from services.ai_engine import chat, generate_christian_content, verify_verse_claim, handle_difficult_theology
from retrieval.scripture_retrival import (
    retrieve_scripture_context,
    validate_book_name
)
from retrieval.scripture import extract_verse_refs, get_denomination_context
from services.image_gen import generate_christian_image, EXAMPLE_IMAGE_PROMPTS
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
    .main-header p  { color: #b8d4e8; margin: 0.3rem 0 0; font-size: 0.95rem; }

    .verse-box {
        background: #f0f7ff;
        border-left: 4px solid #1a3a5c;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-style: italic;
        color: #1a3a5c;
    }
    .retrieved-verse {
        background: #f7f7f2;
        border-left: 3px solid #c8a84b;
        padding: 0.5rem 0.8rem;
        border-radius: 0 6px 6px 0;
        margin: 0.3rem 0;
        font-size: 0.88rem;
        color: #333;
    }
    .grounded-badge {
        background: #d4edda; color: #155724;
        padding: 0.15rem 0.5rem; border-radius: 10px;
        font-size: 0.78rem; font-weight: 600;
        display: inline-block; margin-bottom: 0.4rem;
    }
    .safety-badge-safe    { background:#d4edda; color:#155724; padding:0.2rem 0.6rem; border-radius:12px; font-size:0.8rem; font-weight:600; }
    .safety-badge-block   { background:#f8d7da; color:#721c24; padding:0.2rem 0.6rem; border-radius:12px; font-size:0.8rem; font-weight:600; }
    .safety-badge-caution { background:#fff3cd; color:#856404; padding:0.2rem 0.6rem; border-radius:12px; font-size:0.8rem; font-weight:600; }
    .denom-info {
        background: #fafafa; border: 1px solid #e0e0e0;
        border-radius: 8px; padding: 0.8rem 1rem;
        font-size: 0.88rem; color: #444; margin-top: 0.5rem;
    }
    .stChatMessage { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

defaults = {
    "messages":       [],
    "denomination":   "general",
    "caution_mode":   False,
    "moderation_log": [],
    "image_prompt":   "",
    "tool_mode":      "💬 Chat",
    "pending_input":  None,
    "image_result":   None,
    "pending_image":  False,
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
        ["💬 Chat", "✍️ Content Generator", "🖼️ Image Generator",
         "🔍 Verse Verifier", "🧪 Eval Dashboard"],
        label_visibility="collapsed",
        key="tool_mode",
    )

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages   = []
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
    st.caption("Chat: gpt-4o-mini · Images: Stability AI")
    st.caption("RAG: Local Bible dataset · Safety: moderation layer")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>✝️ Faith & Scripture AI</h1>
    <p>Scripture-grounded · Retrieval-first · Hallucination-prevented · Denomination-sensitive</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def log_moderation(user_input: str, level: str, reason: str = ""):
    st.session_state.moderation_log.append({
        "preview": user_input[:60],
        "level":   level,
        "reason":  reason,
    })


# ═════════════════════════════════════════════════════════════════════════════
# MODE: CHAT
# ═════════════════════════════════════════════════════════════════════════════

if st.session_state.tool_mode == "💬 Chat":

    for msg in st.session_state.messages:
        avatar = "🙏" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if not st.session_state.messages and not st.session_state.get("pending_input"):
        st.markdown("#### 💡 Try asking...")
        cols = st.columns(2)
        suggestions = [
            "What does John 3:16 really mean?",
            "Explain the Trinity in simple terms",
            "What does the Bible say about anxiety?",
            "Why did God allow suffering in Job's life?",
            "Hezekiah 4:11 says trust in the Lord — explain this",
        ]
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_input = s
                st.rerun()

    typed_input = st.chat_input("Ask anything about Christianity, scripture, theology...")
    user_input  = typed_input or st.session_state.pop("pending_input", None)

    if user_input:
        mod = moderate_message(user_input)
        log_moderation(user_input, mod.level.value, mod.reason or "")

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        if mod.level == RiskLevel.BLOCK:
            blocked_reply = f"🛡️ **Content Policy** | {mod.suggested_response}"
            st.session_state.messages.append({"role": "assistant", "content": blocked_reply})
            with st.chat_message("assistant", avatar="🙏"):
                st.markdown(blocked_reply)

        else:
            if mod.level == RiskLevel.CAUTION:
                st.session_state.caution_mode = True

            # Retrieve scripture BEFORE calling LLM — show user what was found
            retrieved_verses = retrieve_scripture_context(user_input, top_k=4)

            difficult_keywords = ["suffering", "evil", "genocide", "homosexual",
                                  "hell", "theodicy", "why did god", "purgatory",
                                  "predestination", "free will"]
            is_difficult = any(kw in user_input.lower() for kw in difficult_keywords)

            with st.chat_message("assistant", avatar="🙏"):

                # Show retrieved verses before response
                if retrieved_verses:
                    st.markdown('<span class="grounded-badge">📖 Scripture-grounded response</span>',
                                unsafe_allow_html=True)
                    with st.expander(
                        f"📚 {len(retrieved_verses)} verse(s) retrieved from local dataset",
                        expanded=False
                    ):
                        for v in retrieved_verses:
                            st.markdown(
                                f'<div class="retrieved-verse">'
                                f'<strong>{v["reference"]}</strong><br>'
                                f'"{v["text"]}"'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            st.caption(
                                f"[BibleGateway ↗](https://www.biblegateway.com/passage/?search="
                                f"{v['reference'].replace(' ', '+')})"
                            )

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
                            f"⚠️ Error: {str(e)}\n\n"
                            "Please ensure your OPENAI_API_KEY is set in your .env file."
                        )

                if st.session_state.caution_mode:
                    st.caption("🟡 Sensitive topic — responding with extra pastoral care")

                st.markdown(response)

                refs = extract_verse_refs(response)
                if refs:
                    with st.expander(
                        f"📖 Scripture references cited ({len(refs)})", expanded=False
                    ):
                        for ref in refs:
                            st.markdown(
                                f"• **{ref}** — "
                                f"[Look up on BibleGateway]"
                                f"(https://www.biblegateway.com/passage/?search={ref.replace(' ', '+')})"
                            )

            st.session_state.messages.append({"role": "assistant", "content": response})


# ═════════════════════════════════════════════════════════════════════════════
# MODE: CONTENT GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.tool_mode == "✍️ Content Generator":

    st.markdown("### ✍️ Christian Content Generator")
    st.caption("Generate prayers, devotionals, sermon outlines — all scripture-grounded via retrieval")

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
                # Show retrieved verses first
                retrieved = retrieve_scripture_context(topic, top_k=3)
                if retrieved:
                    st.markdown('<span class="grounded-badge">📖 Scripture-grounded</span>',
                                unsafe_allow_html=True)
                    with st.expander("📚 Scripture retrieved for grounding", expanded=True):
                        for v in retrieved:
                            st.markdown(
                                f'<div class="retrieved-verse">'
                                f'<strong>{v["reference"]}</strong>: "{v["text"]}"'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                with st.spinner(f"Crafting your {content_type}..."):
                    try:
                        result = generate_christian_content(
                            content_type=content_type,
                            topic=topic,
                            denomination=st.session_state.denomination,
                            tone=tone,
                        )
                        st.markdown(f'<div class="verse-box">{result}</div>',
                                    unsafe_allow_html=True)
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
    st.caption("Generate reverent Christian artwork · Stability AI primary · Pollinations fallback")

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
                st.session_state.pending_image = True
                st.rerun()

        gen_btn = st.button("🎨 Generate Image", type="primary", use_container_width=True)

    should_generate = gen_btn or st.session_state.get("pending_image", False)
    if should_generate:
        st.session_state.pending_image = False

    with col2:
        current_prompt = st.session_state.image_prompt

        if should_generate and current_prompt:
            mod = moderate_image_prompt(current_prompt)
            log_moderation(current_prompt, mod.level.value, mod.reason or "")

            if mod.level == RiskLevel.BLOCK:
                st.error(f"🛡️ {mod.suggested_response}")
                st.session_state.image_result = None
            else:
                with st.spinner("Creating your Christian artwork..."):
                    try:
                        result = generate_christian_image(
                            current_prompt,
                            denomination=st.session_state.denomination,
                        )
                        if result["success"]:
                            if result.get("image_bytes"):
                                st.session_state.image_result = {
                                    "bytes":  result["image_bytes"],
                                    "prompt": result["prompt_used"],
                                }
                            elif result.get("image_url"):
                                st.session_state.image_result = {
                                    "url":    result["image_url"],
                                    "prompt": result["prompt_used"],
                                }
                        else:
                            st.session_state.image_result = {
                                "error": result.get("reason", "Unknown error."),
                            }
                    except Exception as e:
                        st.session_state.image_result = {"error": str(e)}

        elif should_generate and not current_prompt:
            st.warning("Please describe the image you want.")

        result = st.session_state.get("image_result")
        if result:
            if "error" in result:
                st.error(f"Image generation error: {result['error']}")
            elif "bytes" in result:
                st.image(result["bytes"], caption="Generated Christian Artwork",
                         use_container_width=True)
                with st.expander("🎨 Art direction prompt used"):
                    st.caption(result["prompt"])
            elif "url" in result:
                st.image(result["url"], caption="Generated Christian Artwork",
                         use_container_width=True)
                with st.expander("🎨 Art direction prompt used"):
                    st.caption(result["prompt"])
        elif not should_generate:
            st.info("👈 Describe a scene and click Generate")


# ═════════════════════════════════════════════════════════════════════════════
# MODE: VERSE VERIFIER
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.tool_mode == "🔍 Verse Verifier":

    st.markdown("### 🔍 Bible Verse Verifier")
    st.caption("Checks local dataset first · Falls back to LLM with temperature=0 · Never fabricates")

    col1, col2 = st.columns([1, 1])

    with col1:
        verse_ref  = st.text_input("Verse reference", placeholder="e.g. John 3:16")
        verse_text = st.text_area("Verse text to verify",
                                   placeholder="Paste the verse text here...", height=100)
        verify_btn = st.button("🔍 Verify", type="primary")

        st.divider()
        st.markdown("**Quick book validation:**")
        book_check  = st.text_input("Book name", placeholder="e.g. Hezekiah")
        chap_check  = st.number_input("Chapter", min_value=1, max_value=200, value=1)
        verse_check = st.number_input("Verse",   min_value=1, max_value=200, value=1)
        struct_btn  = st.button("Check book")

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
                        st.warning("⚠️ Could not verify definitively")

                    if result.get("text_accurate") is True:
                        st.success("✅ Text is accurate")
                    elif result.get("text_accurate") is False:
                        st.error("❌ Text does not match this verse")

                    if result.get("actual_text"):
                        st.markdown("**Actual verse text (from local dataset or verified source):**")
                        st.markdown(
                            f'<div class="verse-box">{result["actual_text"]}</div>',
                            unsafe_allow_html=True,
                        )

                    if result.get("correct_reference") and result["correct_reference"] != verse_ref:
                        st.info(f"📖 Correct reference: **{result['correct_reference']}**")

                    if result.get("notes"):
                        st.caption(f"Note: {result['notes']}")

                except Exception as e:
                    st.error(f"Verification error: {e}")

        if struct_btn and book_check:
            valid, msg = validate_book_name(book_check)
            if valid:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")


# ═════════════════════════════════════════════════════════════════════════════
# MODE: EVAL DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

elif st.session_state.tool_mode == "🧪 Eval Dashboard":

    st.markdown("### 🧪 Evaluation Dashboard")
    st.caption("Covers hallucination, adversarial prompts, edge cases, and image safety")

    tab1, tab2, tab3 = st.tabs(["📋 Dataset", "🏃 Run Eval", "📊 Architecture"])

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
            for detail in results["details"]:
                icon = "✅" if detail["passed"] else "❌"
                st.markdown(
                    f"{icon} **{detail['id']}** — "
                    f"`{detail['input_preview']}` → `{detail['result']}`"
                )

    with tab3:
        st.markdown("### System Architecture")
        st.markdown("""
```
User Input
    ↓
Moderation Layer          ← regex patterns, instant, no API call
    ↓
Fake Reference Detector   ← validate_book_name() against full book list
    ↓
Scripture Retrieval       ← local bible.json dataset (50 key verses)
    │                        keyword + topic scoring, no LLM needed
    ↓
Prompt Assembly           ← retrieved verses injected into system prompt
    │                        LLM instructed: use ONLY these verses
    ↓
gpt-4o-mini               ← explains/discusses grounded in real text
    ↓
Post-processing           ← extract cited references → BibleGateway links
    ↓
Response
```

**Hallucination Prevention (3 layers)**
1. Fake book detection — "Hezekiah 4:11" blocked before LLM call
2. Local dataset grounding — LLM given actual verse text, told not to invent
3. Verse verifier — checks local dataset first, then LLM at temperature=0

**Image Pipeline**
```
User Prompt → Image Moderation → gpt-4o-mini Prompt Rewriter
    → Stability AI (primary) → Pollinations (fallback) → Display
```

**Safety Layers**
1. Pre-flight regex moderation (instant)
2. Grounded system prompt on every call
3. Difficult theology router (suffering, theodicy, etc.)
4. Separate image safety check
5. Post-generation verse validation + external links

**Denomination Support**
7 traditions: General, Protestant, Catholic, Orthodox, Reformed, Baptist, Lutheran
Deuterocanonical books included for Catholic/Orthodox context
        """)