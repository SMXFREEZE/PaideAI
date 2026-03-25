"""
Professeur Sami - Adaptive Math Tutor
======================================
Run with:  streamlit run app.py

Features:
  - Conversational Socratic tutoring via Claude Sonnet
  - Upload handwritten notes (PDF/image) -> Claude Vision OCR -> RAG
  - SM-2 spaced repetition mastery tracker
  - /quiz, /review, /explain, /hint, /mastery commands
  - Auto micro-quiz every 15 minutes
  - Visual mastery dashboard
  - LaTeX rendering (native in Streamlit markdown)
"""

import os
import sys
import time
from pathlib import Path

# Load .env before any imports that need API key
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Validate API key early
if not os.environ.get("ANTHROPIC_API_KEY"):
    import streamlit as st
    st.error(
        "**ANTHROPIC_API_KEY not set.**\n\n"
        "1. Copy `.env.example` to `.env`\n"
        "2. Add your key: `ANTHROPIC_API_KEY=sk-ant-...`\n"
        "3. Get a key at https://console.anthropic.com"
    )
    st.stop()

import streamlit as st

from src.memory.mastery_tracker import MasteryTracker
from src.memory.note_ingestion import ingest_uploaded_file
from src.memory.vector_store import NoteVectorStore
from src.tutor.session_manager import SessionManager
from src.tutor.socratic_engine import SocraticEngine


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Professeur Sami",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS: clean, academic look with warm accents
# ------------------------------------------------------------------

st.markdown("""
<style>
  /* Chat messages */
  .user-bubble {
    background: #1e3a5f;
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
  }
  .tutor-bubble {
    background: #f0f4ff;
    color: #1a1a2e;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 90%;
    border-left: 4px solid #4a90d9;
  }
  /* Command pill badges */
  .cmd-badge {
    display: inline-block;
    background: #4a90d9;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.85em;
    margin: 2px;
    cursor: pointer;
  }
  /* Micro-quiz alert */
  .quiz-alert {
    background: linear-gradient(135deg, #ff6b35, #f7c59f);
    color: #1a1a2e;
    padding: 12px 20px;
    border-radius: 12px;
    font-weight: bold;
    margin-bottom: 12px;
  }
  /* Mastery bar */
  .mastery-bar-container {
    background: #e0e0e0;
    border-radius: 8px;
    height: 12px;
    margin: 4px 0;
  }
  /* Timer */
  .timer-text {
    font-size: 0.8em;
    color: #666;
    text-align: right;
  }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Session-level singletons (cached across reruns)
# ------------------------------------------------------------------

@st.cache_resource
def get_session() -> SessionManager:
    return SessionManager()

@st.cache_resource
def get_tracker() -> MasteryTracker:
    return MasteryTracker()

@st.cache_resource
def get_store() -> NoteVectorStore:
    return NoteVectorStore()

@st.cache_resource
def get_engine() -> SocraticEngine:
    return SocraticEngine(get_session(), get_tracker(), get_store())


session = get_session()
tracker = get_tracker()
store = get_store()
engine = get_engine()


# ------------------------------------------------------------------
# Streamlit state init
# ------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content}]
if "ingesting" not in st.session_state:
    st.session_state.ingesting = False
if "student_name_set" not in st.session_state:
    st.session_state.student_name_set = False


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.title("🎓 Professeur Sami")
    st.caption("Your adaptive math tutor")
    st.divider()

    # Student name
    st.subheader("Student Profile")
    name = st.text_input("Your name", value=session.state.student_name, key="student_name_input")
    if name and name != session.state.student_name:
        session.state.student_name = name

    interests_input = st.text_input(
        "Your interests (for analogies)",
        value=", ".join(session.state.interests) or "music, programming, basketball",
        help="Professeur Sami will use these to explain concepts",
    )
    if interests_input:
        session.set_interests([i.strip() for i in interests_input.split(",")])

    st.divider()

    # Note upload
    st.subheader("Upload Notes")
    st.caption("PDF, image, or .txt — Claude will OCR handwritten notes")
    uploaded = st.file_uploader(
        "Drop your notes here",
        type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "md"],
        label_visibility="collapsed",
    )
    if uploaded is not None:
        note_label = st.text_input("Label for these notes", value=uploaded.name.split(".")[0])
        if st.button("Ingest Notes", type="primary", use_container_width=True):
            with st.spinner(f"Claude is reading '{uploaded.name}'..."):
                try:
                    chunks = ingest_uploaded_file(
                        uploaded.getvalue(),
                        uploaded.name,
                        note_label=note_label,
                    )
                    added = store.add_chunks(chunks, uploaded.name)
                    st.success(f"Added {added} note chunks to memory!")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    # Existing notes
    labels = store.get_note_labels()
    if labels:
        st.caption(f"Notes in memory: {store.count()} chunks")
        for label in labels:
            col1, col2 = st.columns([3, 1])
            col1.write(f"- {label}")
            if col2.button("X", key=f"del_{label}", help="Delete"):
                store.delete_by_label(label)
                st.rerun()

    st.divider()

    # Session info
    st.subheader("Session")
    elapsed = session.elapsed_minutes()
    st.write(f"Time: {elapsed:.0f} min")
    secs = session.seconds_until_next_quiz()
    st.write(f"Next micro-quiz in: {secs // 60}m {secs % 60}s")
    st.write(f"Quizzes taken: {session.state.quiz_count}")

    st.divider()

    # Mastery dashboard
    st.subheader("Mastery")
    all_mastery = tracker.get_all_mastery()
    if all_mastery:
        for skill_id, mastery in sorted(all_mastery.items(), key=lambda x: x[1]):
            pct = int(mastery * 100)
            color = "#e74c3c" if pct < 40 else "#f39c12" if pct < 70 else "#27ae60"
            st.markdown(
                f"**{skill_id.replace('_', ' ').title()}**",
                help=f"{pct}% mastery",
            )
            st.progress(mastery, text=f"{pct}%")
    else:
        st.caption("No mastery data yet. Take a /quiz to start tracking!")

    st.divider()
    st.caption("Commands: /quiz, /review, /explain [topic], /hint, /mastery")


# ------------------------------------------------------------------
# Main chat area
# ------------------------------------------------------------------

st.header("Professeur Sami", divider="blue")

# Quick command buttons
cols = st.columns(5)
cmds = ["/quiz", "/review", "/explain", "/hint", "/mastery"]
icons = ["🧪", "🔍", "💡", "🎯", "📊"]
for col, cmd, icon in zip(cols, cmds, icons):
    if col.button(f"{icon} {cmd.strip('/')}", use_container_width=True):
        if cmd == "/explain":
            st.session_state["pending_command"] = None
            st.session_state["show_explain_input"] = True
        else:
            st.session_state["pending_command"] = cmd

# Explain concept input
if st.session_state.get("show_explain_input"):
    concept = st.text_input("What concept to explain?", key="explain_concept_input")
    if st.button("Explain it", type="primary"):
        if concept:
            st.session_state["pending_command"] = f"/explain {concept}"
            st.session_state["show_explain_input"] = False
            st.rerun()

# Micro-quiz alert
if session.check_micro_quiz_due() and len(st.session_state.messages) > 4:
    st.markdown(
        '<div class="quiz-alert">⏰ 15 minutes have passed! Time for a quick micro-quiz.</div>',
        unsafe_allow_html=True,
    )
    if st.button("Take Micro-Quiz Now", type="primary"):
        st.session_state["pending_command"] = "__micro_quiz__"
    if st.button("Skip this time"):
        session.mark_quiz_done()
        st.rerun()

# Render conversation history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🎓"):
                st.markdown(msg["content"])


# ------------------------------------------------------------------
# Handle pending commands (from button clicks)
# ------------------------------------------------------------------

def run_command_or_chat(text: str, mode: str = "tutor") -> None:
    """Stream response and append to message history."""
    st.session_state.messages.append({"role": "user", "content": text})

    with st.chat_message("user"):
        st.markdown(text)

    with st.chat_message("assistant", avatar="🎓"):
        placeholder = st.empty()
        full_text = ""

        if text.startswith("/") or mode != "tutor":
            stream = engine.handle_command(text) if text.startswith("/") else engine.chat_stream(text, mode)
        else:
            stream = engine.chat_stream(text)

        for chunk in stream:
            full_text += chunk
            placeholder.markdown(full_text + "▌")

        placeholder.markdown(full_text)

    st.session_state.messages.append({"role": "assistant", "content": full_text})


pending = st.session_state.pop("pending_command", None)
if pending:
    if pending == "__micro_quiz__":
        st.session_state.messages.append({"role": "user", "content": "⏰ Micro-quiz time!"})
        with st.chat_message("assistant", avatar="🎓"):
            placeholder = st.empty()
            full_text = ""
            for chunk in engine.generate_micro_quiz():
                full_text += chunk
                placeholder.markdown(full_text + "▌")
            placeholder.markdown(full_text)
        st.session_state.messages.append({"role": "assistant", "content": full_text})
    else:
        run_command_or_chat(pending)
    st.rerun()


# ------------------------------------------------------------------
# Chat input
# ------------------------------------------------------------------

if prompt := st.chat_input("Ask a math question, or type /quiz, /review, /explain [topic]..."):
    is_command = prompt.strip().startswith("/")
    run_command_or_chat(prompt)
    st.rerun()


# ------------------------------------------------------------------
# First-time welcome message
# ------------------------------------------------------------------

if not st.session_state.messages:
    student = session.state.student_name
    welcome = f"""Bonjour **{student}**! I'm **Professeur Sami**, your personal adaptive math tutor.

Here's how I work:
- Ask me any math question and I'll guide you with the **Socratic method** (hints, not answers)
- Upload your **handwritten notes** in the sidebar: I'll read them and use them to teach you
- Every **15 minutes**, I'll give you a **micro-quiz** to lock in what you've learned
- Use commands: **/quiz**, **/review**, **/explain [concept]**, **/hint**, **/mastery**

I track your mastery of each skill using **spaced repetition** so you review what you're weak on, not what you already know.

**To get started:** What topic are you working on today?"""

    with st.chat_message("assistant", avatar="🎓"):
        st.markdown(welcome)
    st.session_state.messages.append({"role": "assistant", "content": welcome})
    session.add_message("assistant", welcome)
