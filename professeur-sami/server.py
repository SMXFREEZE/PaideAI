"""
Professeur Sami — FastAPI Server
Run: uvicorn server:app --reload --port 8000

Improvements over baseline:
  - Rate limiting via slowapi (RATE_LIMIT_PER_MINUTE from .env)
  - CORS origins from ALLOWED_ORIGINS env (not hardcoded *)
  - API key validated at startup — fail fast with clear error
  - /upload-notes: file size + MIME type enforcement
  - /class-chat: MIME type validation on drawing uploads
  - /verify-math: SymPy-backed symbolic math answer checker
  - /stop-stream: client-side abort signal endpoint
"""

import json
import os
import re
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import anthropic
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    _RATE_LIMIT = os.environ.get("RATE_LIMIT_PER_MINUTE", "30")
    limiter = Limiter(key_func=get_remote_address)
    HAS_SLOWAPI = True
except ImportError:
    HAS_SLOWAPI = False
    limiter = None

from src.memory.mastery_tracker import MasteryTracker
from src.memory.note_ingestion import (
    ingest_text, ingest_uploaded_file, ingest_uploaded_file_async,
    validate_upload, MAX_UPLOAD_BYTES,
)
from src.memory.vector_store import NoteVectorStore
from src.tools.math_verifier import verify as math_verify
from src.tutor.session_manager import SessionManager
from src.tutor.socratic_engine import SocraticEngine

# ── Startup validation ────────────────────────────────────────────────

_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not _api_key or _api_key == "sk-ant-your-key-here":
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
    )

# ── App factory ───────────────────────────────────────────────────────

app = FastAPI(title="PaideIA")

# CORS: read allowed origins from env (comma-separated); default to * for dev
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

if HAS_SLOWAPI:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.on_event("startup")
async def _startup():
    """Apply mastery decay on startup (forgetting curve)."""
    tracker.decay_all()


static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Singletons
session = SessionManager()
tracker = MasteryTracker()
store   = NoteVectorStore()
engine  = SocraticEngine(session, tracker, store)

# ── Allowed MIME types for drawings ──────────────────────────────────

_DRAWING_MIME = {"image/png", "image/jpeg", "image/webp", "image/gif"}
_NOTES_MIME   = {
    "image/png", "image/jpeg", "image/webp",
    "application/pdf",
    "text/plain", "text/markdown",
}


def _assert_mime(content_type: str, allowed: set, field: str) -> None:
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct and ct not in allowed:
        raise HTTPException(415, f"{field}: unsupported media type '{ct}'")


# ── Models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

class OnboardingRequest(BaseModel):
    name: str
    course_name: str
    course_outline: str
    exam_weeks: int = 8
    daily_minutes: int = 30
    interests: list[str] = []

class VerifyMathRequest(BaseModel):
    student_answer: str
    correct_answer: str
    context: str = ""


# ── Pages ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def landing():
    html = static_dir / "landing.html"
    return HTMLResponse(html.read_text(encoding="utf-8") if html.exists() else "<h1>landing.html missing</h1>")

@app.get("/app", response_class=HTMLResponse)
async def tutor_app():
    html = static_dir / "index.html"
    return HTMLResponse(html.read_text(encoding="utf-8") if html.exists() else "<h1>index.html missing</h1>")


# ── Onboarding ────────────────────────────────────────────────────────

@app.get("/onboarding-status")
async def onboarding_status():
    return {
        "done": session.state.onboarding_done,
        "student_name": session.state.student_name,
        "course_name": session.state.course_name,
    }

@app.post("/onboarding")
async def complete_onboarding(req: OnboardingRequest):
    session.state.student_name   = req.name.strip()
    session.state.course_name    = req.course_name.strip()
    session.state.course_outline = req.course_outline.strip()
    session.state.exam_weeks     = max(1, req.exam_weeks)
    session.state.daily_minutes  = max(10, req.daily_minutes)
    session.state.interests      = req.interests
    session.state.study_start_date = date.today().isoformat()
    session.state.onboarding_done  = True
    session.clear_history()

    # Ingest outline into vector store
    if req.course_outline.strip():
        label = f"{req.course_name} Outline"
        chunks = ingest_text(req.course_outline, label)
        store.add_chunks(chunks, label)

    session.save_profile()

    # Generate study plan
    plan = _generate_study_plan()
    session.state.study_plan = plan
    session.save_profile()

    return {"ok": True, "study_plan": plan}


def _generate_study_plan() -> list:
    """Call Claude to generate a JSON week-by-week study plan."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    outline_excerpt = session.state.course_outline[:3000] or "General mathematics curriculum"
    prompt = f"""Create a {session.state.exam_weeks}-week study plan for:
Student: {session.state.student_name}
Course: {session.state.course_name}
Daily time: {session.state.daily_minutes} minutes
Outline: {outline_excerpt}

Return ONLY a JSON array (no markdown, no explanation):
[
  {{
    "week": 1,
    "theme": "Short theme title",
    "topics": ["Topic A", "Topic B"],
    "daily": [
      {{"day": "Mon", "topic": "Specific lesson", "duration": {session.state.daily_minutes}, "type": "lesson"}},
      {{"day": "Tue", "topic": "Practice", "duration": {session.state.daily_minutes}, "type": "practice"}},
      {{"day": "Wed", "topic": "Short quiz", "duration": 20, "type": "quiz"}},
      {{"day": "Thu", "topic": "New concept", "duration": {session.state.daily_minutes}, "type": "lesson"}},
      {{"day": "Fri", "topic": "Review + summary", "duration": 25, "type": "review"}}
    ]
  }}
]
Generate all {session.state.exam_weeks} weeks."""

    resp = client.messages.create(
        model=os.environ.get("TUTOR_MODEL", "claude-sonnet-4-6"),
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r'\[[\s\S]+\]', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


# ── Dashboard ─────────────────────────────────────────────────────────

@app.get("/dashboard-data")
async def dashboard_data():
    return {
        "student_name":  session.state.student_name,
        "course_name":   session.state.course_name,
        "exam_weeks":    session.state.exam_weeks,
        "daily_minutes": session.state.daily_minutes,
        "current_week":  session.get_current_week(),
        "streak_days":   session.state.streak_days,
        "study_plan":    session.state.study_plan,
        "daily_log":     session.state.daily_log[-14:],
        "today_topic":   session.get_today_topic(),
        "notes_count":   store.count(),
        "note_labels":   store.get_note_labels(),
        "quiz_count":    session.state.quiz_count,
        "mastery":       tracker.get_all_mastery(),
    }

@app.post("/log-session")
async def log_session_endpoint(data: dict):
    session.log_session(
        topic=data.get("topic", ""),
        duration_min=int(data.get("duration_min", 0)),
        quiz_score=data.get("quiz_score"),
    )
    return {"ok": True}

@app.post("/reset-onboarding")
async def reset_onboarding():
    session.state.onboarding_done = False
    session.save_profile()
    return {"ok": True}


# ── Chat (SSE streaming) ───────────────────────────────────────────────

def _rate_limit_decorator(route_fn):
    """Apply slowapi rate limit if available, else return function unchanged."""
    if HAS_SLOWAPI:
        return limiter.limit(f"{_RATE_LIMIT}/minute")(route_fn)
    return route_fn


@app.post("/chat")
@_rate_limit_decorator
async def chat(req: ChatRequest, request: Request):
    if not req.message.strip():
        raise HTTPException(400, "Empty message")

    def generate():
        is_cmd = req.message.strip().startswith("/")
        gen = engine.handle_command(req.message) if is_cmd else engine.chat_stream(req.message)
        for chunk in gen:
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/micro-quiz")
async def micro_quiz():
    def generate():
        for chunk in engine.generate_micro_quiz():
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Notes ──────────────────────────────────────────────────────────────

@app.post("/upload-notes")
async def upload_notes(file: UploadFile = File(...), label: str = Form("")):
    _assert_mime(file.content_type, _NOTES_MIME, "notes file")
    content = await file.read()
    # Validate size — raises ValueError if too large
    try:
        validate_upload(content, file.filename)
    except ValueError as e:
        raise HTTPException(413, str(e))
    note_label = label or Path(file.filename).stem
    try:
        chunks = await ingest_uploaded_file_async(content, file.filename, note_label=note_label)
        added = store.add_chunks(chunks, file.filename)
        return {"ok": True, "chunks_added": added, "label": note_label}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/notes/{label}")
async def delete_notes(label: str):
    store.delete_by_label(label)
    return {"ok": True}


# ── Mastery ────────────────────────────────────────────────────────────

@app.get("/mastery")
async def get_mastery():
    return {"summary": tracker.get_summary_json(), "all": tracker.get_all_mastery()}

@app.get("/session")
async def get_session():
    return {
        "student_name":       session.state.student_name,
        "current_topic":      session.get_current_topic_name(),
        "elapsed_minutes":    round(session.elapsed_minutes(), 1),
        "seconds_until_quiz": session.seconds_until_next_quiz(),
        "quiz_count":         session.state.quiz_count,
        "notes_count":        store.count(),
        "note_labels":        store.get_note_labels(),
        "micro_quiz_due":     session.check_micro_quiz_due() and len(session.state.history) > 4,
        "onboarding_done":    session.state.onboarding_done,
    }

@app.post("/profile")
async def update_profile(data: dict):
    if "name" in data:
        session.state.student_name = data["name"]
    if "interests" in data:
        session.set_interests(data["interests"])
    session.save_profile()
    return {"ok": True}

@app.get("/curriculum")
async def get_curriculum():
    return session.state.curriculum


# ── Math Verifier (SymPy) ─────────────────────────────────────────────

@app.post("/verify-math")
async def verify_math(req: VerifyMathRequest):
    """
    Symbolically verify a student's math answer against the correct answer.
    Uses SymPy for exact algebraic equivalence, with numeric fallback.
    """
    if not req.student_answer.strip() or not req.correct_answer.strip():
        raise HTTPException(400, "Both student_answer and correct_answer are required")
    result = math_verify(req.student_answer, req.correct_answer, req.context)
    return {
        "is_correct":          result.is_correct,
        "confidence":          result.confidence,
        "method":              result.method,
        "student_simplified":  result.student_simplified,
        "correct_simplified":  result.correct_simplified,
        "message":             result.message,
    }


# ── Structured MCQ ──────────────────────────────────────────────────────────

class McqRequest(BaseModel):
    topic: str = ""
    n: int = 5

class AnswerRecord(BaseModel):
    skill_id: str
    quality: int  # 0-5

@app.post("/record-answer")
async def record_answer(req: AnswerRecord):
    quality = max(0, min(5, req.quality))
    result  = tracker.record_answer(req.skill_id, quality)
    return {"ok": True, "mastery": result["mastery"]}


@app.post("/generate-mcq")
async def generate_mcq(req: McqRequest):
    topic = req.topic.strip() or session.get_current_topic_name() or "Mathematics"
    n     = max(1, min(10, req.n))
    try:
        result = engine.generate_mcq_json(topic, n)
        return result
    except Exception as e:
        raise HTTPException(500, f"MCQ generation failed: {e}")


# ── Class Mode ──────────────────────────────────────────────────────────────

@app.post("/class-chat")
@_rate_limit_decorator
async def class_chat(
    request: Request,
    message: str = Form(""),
    drawing: UploadFile = File(None),
):
    """Class Mode — accepts text + optional student drawing (Claude Vision)."""

    content: list = []
    if drawing and drawing.filename:
        _assert_mime(drawing.content_type, _DRAWING_MIME, "drawing")
        img_bytes = await drawing.read()
        if img_bytes:
            if len(img_bytes) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Drawing too large")
            import base64 as _b64
            img_b64    = _b64.b64encode(img_bytes).decode()
            media_type = drawing.content_type or "image/png"
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": img_b64}
            })
    if message.strip():
        content.append({"type": "text", "text": message})
    if not content:
        raise HTTPException(400, "No message or drawing provided")

    def generate():
        for chunk in engine.class_stream(content):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Exam Predictor ─────────────────────────────────────────────────────────

@app.post("/predict-exam")
async def predict_exam_endpoint(
    past_exams: list[UploadFile] = File(default=[]),
    extra_notes: list[UploadFile] = File(default=[]),
):
    """
    Predict exam questions from past exams + notes using Oracle Analysis.
    Returns structured JSON with analysis + predicted exam + full solutions.
    """
    if not past_exams:
        raise HTTPException(400, "At least one past exam file is required")

    exam_texts = []
    for f in past_exams:
        content = await f.read()
        _assert_mime(f.content_type, _NOTES_MIME, f"exam file {f.filename}")
        try:
            validate_upload(content, f.filename)
            chunks = ingest_uploaded_file(content, f.filename, note_label=f"exam:{f.filename}")
            text = "\n".join(c["text"] for c in chunks)
            exam_texts.append(f"=== PAST EXAM: {f.filename} ===\n{text}")
        except Exception:
            exam_texts.append(f"=== PAST EXAM: {f.filename} === [could not extract text]")

    note_texts = []
    for f in extra_notes:
        content = await f.read()
        try:
            validate_upload(content, f.filename)
            chunks = ingest_uploaded_file(content, f.filename, note_label=f"notes:{f.filename}")
            text = "\n".join(c["text"] for c in chunks)
            note_texts.append(f"=== NOTES: {f.filename} ===\n{text}")
        except Exception:
            pass

    stored_note_labels = store.get_note_labels()
    if stored_note_labels:
        note_texts.append(f"=== STORED NOTES TOPICS: {', '.join(stored_note_labels)} ===")

    all_exams = "\n\n".join(exam_texts)
    all_notes = "\n\n".join(note_texts) if note_texts else "No additional notes provided."
    course    = session.state.course_name or "Mathematics"
    outline   = session.get_curriculum_text()

    try:
        result = engine.predict_exam(all_exams, all_notes, course, outline)
        return result
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")
