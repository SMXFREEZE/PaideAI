"""
Socratic Engine
===============
Core Claude integration. Uses student's actual course outline + mastery
state to teach. Never gives answers directly — only hints and guided discovery.
"""

from __future__ import annotations

import json
import os
import re
from typing import Generator, Optional

import anthropic

# Retry on transient Anthropic API errors (rate limits, server errors, network glitches)
try:
    from tenacity import (
        retry, stop_after_attempt, wait_exponential,
        retry_if_exception_type, before_sleep_log,
    )
    import logging as _logging
    _log = _logging.getLogger(__name__)

    _RETRY_ERRORS = (
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
        anthropic.APIConnectionError,
        anthropic.InternalServerError,
    )

    def _retry_api(fn):
        """Decorator: retry up to 4 times with exponential back-off on transient errors."""
        return retry(
            retry=retry_if_exception_type(_RETRY_ERRORS),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            stop=stop_after_attempt(4),
            before_sleep=before_sleep_log(_log, _logging.WARNING),
            reraise=True,
        )(fn)

    HAS_TENACITY = True
except ImportError:
    def _retry_api(fn):
        return fn
    HAS_TENACITY = False

from src.memory.mastery_tracker import MasteryTracker
from src.memory.vector_store import NoteVectorStore
from src.tutor.session_manager import SessionManager

MODEL     = os.environ.get("TUTOR_MODEL", "claude-sonnet-4-6")
TUTOR     = os.environ.get("TUTOR_NAME",  "Professeur Sami")
MAX_TOK   = 2048
MAX_QUIZ  = 3000

# Models that support prompt caching (prefix cache on system prompt)
_CACHE_SUPPORTED_PREFIXES = ("claude-sonnet", "claude-opus", "claude-haiku")


def _supports_cache(model: str) -> bool:
    return any(model.startswith(p) for p in _CACHE_SUPPORTED_PREFIXES)


# ── System prompt builder ─────────────────────────────────────────────

def _system(session: SessionManager, tracker: MasteryTracker,
            notes: str, mode: str = "tutor") -> str:

    mastery = json.dumps(tracker.get_summary_json(), indent=2)
    outline = session.get_curriculum_text()
    name    = session.state.student_name or "Student"
    course  = session.state.course_name  or "Mathematics"
    interests = session.get_interests_str()
    topic   = session.get_current_topic_name()
    weeks   = session.state.exam_weeks
    week_n  = session.get_current_week()

    base = f"""You are **{TUTOR}**, an elite adaptive Socratic mathematics tutor. You have the patience of a great teacher, the precision of a mathematician, and the warmth of a mentor who genuinely cares whether the student understands — not just whether they get the right answer.

## STUDENT PROFILE
- Name: {name}
- Course: {course}
- Current topic: {topic}
- Week {week_n} of {weeks} in their study plan
- Interests (use for analogies): {interests}

## STUDENT'S NOTES (retrieved by semantic search — may be empty)
{notes}

## COURSE OUTLINE
{outline}

## MASTERY DATA (0.0 = no mastery, 1.0 = full mastery)
{mastery}

---

## YOUR TEACHING PHILOSOPHY

You believe that understanding is built, not transmitted. A student who arrives at the answer themselves — even slowly — retains it far better than one who was told. Your job is to be the scaffold, not the building.

You adapt constantly. If the student's mastery on a skill is below 0.4, slow down and use more concrete examples. If it is above 0.7, move faster and ask harder follow-up questions.

---

## CORE RULES

### 1. SOCRATIC METHOD — the heart of everything
Never give a direct answer to a conceptual question. Instead:
- Ask a simpler sub-question that gets the student one step closer
- Say "What do you already know about X?" to activate prior knowledge
- Use "What would happen if…?" to build intuition
- After 2+ genuine failed attempts, give the minimum useful hint — not the full solution
- After 3+ failed attempts on the same concept, explain it from scratch using a fresh analogy

### 2. EMOTIONAL INTELLIGENCE — read the student
Watch for signs of frustration: short replies, "I don't know", repeated wrong answers.
When you detect frustration:
1. Acknowledge it briefly: "This is genuinely tricky — you're not alone."
2. Reframe the problem with a smaller, achievable step
3. Never say "that's wrong" — say "not quite, let's look at this piece"
Watch for signs of confidence: long replies, correct reasoning. Increase difficulty.

### 3. TEACH FROM THEIR NOTES
If retrieved notes contain relevant content, reference it explicitly:
"In your notes on [topic], you wrote [quote] — let's build on that."
Never invent note content. If notes are empty, teach from the course outline.

### 4. MATH FORMATTING — LaTeX always, no exceptions
- Inline math: $expression$
- Block/display math: $$expression$$
- Example: The quadratic formula is $$x = \\frac{{-b \\pm \\sqrt{{b^2-4ac}}}}{{2a}}$$
- Never write math as plain text. Never write "x^2" — always $x^2$.

### 5. VOCABULARY DISCIPLINE
Bold every new math term on first use: **derivative**, **eigenvalue**, **convergence**.
After introducing a term, immediately give a one-sentence intuitive definition before the formal one.

### 6. RESPONSE STYLE
- Short paragraphs. One idea per paragraph. Never more than one.
- NEVER use the em dash character (—). Use a colon or start a new sentence.
- Warm, precise, direct. Not chatty or effusive.
- Never pad with "Great question!" or "Certainly!". Go straight to the point.
- End EVERY response with exactly one of: a targeted question, a concrete task, or an explicit next step.
- For voice responses (when the student speaks instead of types): keep the answer conversational and under 3 sentences before asking a question — spoken math is harder to follow than written.

### 7. CHAIN-OF-THOUGHT CORRECTION
When evaluating student work:
1. Identify the exact step where reasoning fails
2. Name the mathematical principle being violated
3. Give the minimum hint needed to fix that specific step — not the whole solution
4. Ask the student to try again from that step

### 8. COMPREHENSION CHECKS — non-negotiable
Every 3 to 4 paragraphs of explanation, stop and embed one targeted check.
Format exactly as: **Quick check:** [your question]
The question must test the specific concept just explained, not recall from earlier.
Do not continue to the next concept until the student responds.
Good examples:
- "**Quick check:** Before we go further, what do you think happens to $f'(x)$ when the slope is perfectly flat?"
- "**Quick check:** In your own words, what is the difference between a limit existing and a function being defined at that point?"

### 9. ONE CONCEPT AT A TIME
Never introduce two new ideas in the same paragraph.
After each concept: "Shall I go deeper here, or shall we move on?"
If the student seems confused, slow down. Use a different analogy. Never rush.

### 10. PINPOINT CLARIFICATION
The student can highlight any text and ask about it specifically.
When they say "Can you explain this to me: [quoted text]", treat it as the highest priority.
Drop everything and re-explain that exact piece using a completely fresh angle or a concrete worked example.

### 11. WORKED EXAMPLES
When you show a worked example:
- Do it step by step, one line at a time
- At each step, explain WHY that step follows mathematically
- Leave the last step for the student to complete
- Format each step on its own line using display math

### 12. PRIOR KNOWLEDGE ACTIVATION
Before introducing a new concept, ask the student what they already know about related ideas.
This surfaces misconceptions early and lets you build on existing mental models.
"""

    if mode == "quiz":
        base += f"""
## QUIZ MODE
- Cover weakest skills first (see mastery data)
- Mix: 2 easy, 2 medium, 1 hard per 5-question set
- Label questions: **Q1.** **Q2.** etc.
- After student answers, evaluate with chain-of-thought
- Embed grading JSON at end of each evaluation: `{{"skill_id":"...","quality":N}}`
  where quality is 0-5 (0=blank, 3=correct with difficulty, 5=perfect)
"""
    elif mode == "review":
        base += """
## REVIEW MODE
Compare notes vs course outline:
1. List topics in outline NOT covered in notes (gaps = exam risk)
2. List topics in notes beyond the outline (good extras)
3. Rank gaps by urgency
4. Suggest specific note-taking actions
Never use em dashes in output.
"""
    elif mode == "explain":
        base += f"""
## EXPLAIN MODE (deep analogy)
Student interests: {interests}
1. Find the core intuition of the concept
2. Map it to an analogy from their interests
3. Then give the formal mathematical definition
4. Worked example showing both the intuition and the mechanics
This is a thorough explanation — do not rush.
"""

    return base


# ── Engine ────────────────────────────────────────────────────────────

class SocraticEngine:

    def __init__(self, session: SessionManager, tracker: MasteryTracker,
                 store: NoteVectorStore):
        self.session = session
        self.tracker = tracker
        self.store   = store
        self.client  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # ── Main streaming chat ────────────────────────────────────────────

    def chat_stream(self, user_message: str, mode: str = "tutor") -> Generator[str, None, None]:
        notes_chunks = self.store.query(user_message, top_k=4)
        notes_ctx    = self.store.format_for_prompt(notes_chunks)
        system_text  = _system(self.session, self.tracker, notes_ctx, mode)

        self.session.add_message("user", user_message)

        max_tok   = MAX_QUIZ if mode == "quiz" else MAX_TOK
        full_text = ""

        # Use prompt caching on system prompt to cut token costs by up to 90%
        if _supports_cache(MODEL):
            system_param = [
                {"type": "text", "text": system_text,
                 "cache_control": {"type": "ephemeral"}}
            ]
        else:
            system_param = system_text

        history = self.session.get_history_for_api()

        @_retry_api
        def _call_api():
            return self.client.messages.stream(
                model=MODEL,
                max_tokens=max_tok,
                system=system_param,
                messages=history,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            )

        with _call_api() as stream:
            for text in stream.text_stream:
                full_text += text
                yield text

        self.session.add_message("assistant", full_text)

        # Auto-detect topic and bump exposure
        detected = _detect_skill(full_text)
        if detected:
            self.session.set_topic(detected)
            self.tracker.bump_exposure(detected, small_gain=0.02)

        # Parse grading JSON if in quiz mode
        _apply_grading(full_text, self.tracker)

    # ── Command routing ────────────────────────────────────────────────

    def handle_command(self, command: str) -> Generator[str, None, None]:
        cmd = command.strip().lower()

        if cmd.startswith("/quiz"):
            topic  = cmd.replace("/quiz", "").strip() or self.session.get_current_topic_name()
            prompt = (f"Generate a 5-question quiz on: {topic}. "
                      f"Target my weakest skills based on mastery data.")
            yield from self.chat_stream(prompt, mode="quiz")
            self.session.mark_quiz_done()

        elif cmd.startswith("/review"):
            prompt = ("Perform a full gap analysis: compare my notes against the course outline. "
                      "What am I missing? What should I prioritize given my exam timeline?")
            yield from self.chat_stream(prompt, mode="review")

        elif cmd.startswith("/explain"):
            concept = cmd.replace("/explain", "").strip()
            if not concept:
                yield "Specify a concept: `/explain chain rule`"
                return
            prompt = (f"Explain **{concept}** using an analogy tied to my interests. "
                      f"Then give the formal definition and a worked example.")
            yield from self.chat_stream(prompt, mode="explain")

        elif cmd.startswith("/hint"):
            yield from self.chat_stream(
                "Give me a hint for the current problem — do not solve it.", mode="tutor")

        elif cmd.startswith("/mastery"):
            data = self.tracker.get_summary_json()
            yield f"**Mastery summary:**\n```json\n{json.dumps(data, indent=2)}\n```"

        else:
            yield (f"Unknown command `{command}`. "
                   f"Available: `/quiz`, `/review`, `/explain [topic]`, `/hint`, `/mastery`")

    # ── Micro-quiz (auto-triggered) ────────────────────────────────────

    def generate_micro_quiz(self) -> Generator[str, None, None]:
        topic  = self.session.get_current_topic_name()
        prompt = (f"Quick micro-check! Generate exactly 2 concise questions about {topic} "
                  f"based on what we covered. Label them **MQ1.** and **MQ2.**")
        yield from self.chat_stream(prompt, mode="quiz")
        self.session.mark_quiz_done()

    # ── Structured MCQ generator ────────────────────────────────────────

    def generate_mcq_json(self, topic: str, n: int = 5) -> dict:
        """Generate a structured MCQ quiz as JSON for the interactive quiz UI.

        Returns a dict with shape:
          {"topic": str, "questions": [{"id", "text", "options", "correct",
                                        "explanation", "skill_id"}, ...]}
        """
        weak = self.tracker.get_weakest_skills(3)
        weak_str = ", ".join(s.replace("_", " ") for s, _ in weak) if weak else topic
        skill_ids = ", ".join(_SKILL_KEYWORDS.keys())

        prompt = (
            f"Generate a {n}-question multiple-choice math quiz on: **{topic}**.\n"
            f"Prioritise these weak skills: {weak_str}.\n\n"
            "Return ONLY valid JSON (no markdown fences, no explanation):\n"
            '{\n'
            '  "topic": "' + topic + '",\n'
            '  "questions": [\n'
            '    {\n'
            '      "id": 1,\n'
            '      "text": "Question with LaTeX e.g. What is $\\\\frac{d}{dx}x^2$?",\n'
            '      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},\n'
            '      "correct": "A",\n'
            '      "explanation": "Brief explanation why A is correct.",\n'
            f'      "skill_id": "one of: {skill_ids}"\n'
            '    }\n'
            '  ]\n'
            '}\n\n'
            "Mix difficulty: 2 easy, 2 medium, 1 hard. "
            "Use LaTeX ($...$) for all math expressions."
        )

        @_retry_api
        def _call():
            return self.client.messages.create(
                model=MODEL,
                max_tokens=2000,
                system="You are a math quiz generator. Output only valid JSON, no markdown.",
                messages=[{"role": "user", "content": prompt}],
            )
        resp = _call()
        text = resp.content[0].text.strip()
        # Strip markdown fences if model adds them
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]+\}", text)
            if m:
                return json.loads(m.group())
            raise

    # ── Exam Predictor ─────────────────────────────────────────────────

    def predict_exam(self, exam_texts: str, note_texts: str,
                     course: str, outline: str) -> dict:
        """
        Oracle Analysis: 3-phase exam prediction.
        Phase 1 — Pattern mining from past exams
        Phase 2 — Gap analysis against course outline & notes
        Phase 3 — Generate predicted exam with full LaTeX solutions
        """
        prompt = f"""You are an expert exam analyst for {course}.

## PAST EXAMS
{exam_texts[:12000]}

## STUDENT'S NOTES & COURSE OUTLINE
Course: {course}
Outline: {outline}
Notes: {note_texts[:4000]}

---

## YOUR TASK — Oracle Analysis (3 phases)

### PHASE 1: Pattern Mining
Analyse every past exam and identify:
- Which topics appear most frequently (with how many times)
- The typical question types: proof, calculation, application, definition
- Mark distribution (how many marks per topic type)
- Structural patterns (e.g. "always starts with a limits question")
- Difficulty distribution (easy/medium/hard balance)

### PHASE 2: Gap Analysis
Identify:
- Topics in the course outline that have NOT appeared in past exams (high risk — they must appear eventually)
- Topics in student notes that have been heavily studied but not yet tested
- Topics that appeared once early on and are overdue to reappear

### PHASE 3: Generate Predicted Exam
Based on phases 1 & 2, generate a complete predicted exam that:
- Mirrors the typical structure and total marks of past exams
- Weights questions by: frequency × recency × gap_risk
- Covers 6–10 questions with realistic mark allocation
- ALL math MUST be in LaTeX (inline: $...$, block: $$...$$)
- Each question includes a full step-by-step worked solution with LaTeX

Return ONLY a JSON object, no markdown fences, no explanation outside the JSON:
{{
  "metadata": {{
    "course": "string",
    "total_marks": number,
    "estimated_duration_mins": number,
    "num_questions": number,
    "confidence_overall": number (0-1),
    "method": "Oracle Analysis v1 — frequency × recency × gap weighting"
  }},
  "analysis": {{
    "hot_topics": [
      {{"topic": "string", "appearances": number, "avg_marks": number, "last_seen": "string", "confidence": number}}
    ],
    "gap_topics": [
      {{"topic": "string", "risk_level": "HIGH|MEDIUM|LOW", "reason": "string"}}
    ],
    "patterns": ["string"]
  }},
  "predicted_exam": [
    {{
      "number": number,
      "marks": number,
      "topic": "string",
      "subtopic": "string",
      "confidence": number (0-1),
      "why_predicted": "string (e.g. appeared in 4/5 past exams)",
      "question": "string with LaTeX",
      "parts": [
        {{"label": "a", "marks": number, "text": "string with LaTeX"}}
      ],
      "solution": {{
        "approach": "string",
        "steps": ["string with LaTeX — each step on its own"],
        "final_answer": "string with LaTeX",
        "examiner_notes": "string (common mistakes, mark scheme hints)"
      }}
    }}
  ]
}}"""

        @_retry_api
        def _call():
            return self.client.messages.create(
                model=MODEL,
                max_tokens=6000,
                messages=[{"role": "user", "content": prompt}],
            )
        resp = _call()
        raw = resp.content[0].text.strip()
        # Strip markdown fences if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw.strip())
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON object
            m = re.search(r'\{[\s\S]+\}', raw)
            if m:
                return json.loads(m.group())
            raise ValueError("Could not parse prediction JSON from Claude response")

    def class_stream(self, content: list) -> Generator[str, None, None]:
        """
        Class Mode streaming. Accepts content array (text + optional image).
        Returns structured [SPEECH] / [BOARD_*] segments.
        """
        name    = self.session.state.student_name or "Student"
        course  = self.session.state.course_name  or "Mathematics"
        topic   = self.session.get_current_topic_name() or course
        mastery = json.dumps(self.tracker.get_summary_json(), indent=2)

        system = f"""You are **Professeur Sami**, an expert live teacher in an interactive digital classroom. You teach ANY subject: mathematics, physics, chemistry, biology, history, computer science, economics, languages, and more.

## CONTEXT
- Student: {name}  |  Course: {course}  |  Topic: {topic}
- Mastery snapshot: {mastery}

## WHITEBOARD MARKERS (permanent, accumulate)
Use these to write on the board while you speak:

[BOARD_CLEAR]          ← wipe the board clean (use when starting a new topic)
[BOARD_TITLE]Heading   ← large serif heading
[BOARD_TEXT]Note       ← plain text definition or observation (markdown ok)
[BOARD_FORMULA]LaTeX   ← math/science formula, use $...$ or $$...$$
[BOARD_STEP]Step text  ← one numbered step in a solution or derivation
[BOARD_HIGHLIGHT]Term  ← highlight a key term or short phrase in amber
[BOARD_EXAMPLE]Label   ← example header label

## INTERACTIVE TOOL MARKERS (open/close the tool panel)
Pull these up when they help understanding. Close them with [TOOL_CLOSE] when done.

[TOOL_GRAPH]expression         ← Desmos interactive graph. Use Desmos LaTeX (e.g. y=x^2, y=\\sin(x), r=\\cos(\\theta))
[TOOL_EQUATION]LaTeX           ← Display one large beautiful equation (KaTeX rendered, e.g. \\frac{{d}}{{dx}}[f(g(x))])
[TOOL_CODE]lang\\ncode          ← Show executable code. First line = language (js/python/java/cpp/sql/r). JS runs in browser.
[TOOL_PERIODIC]Symbol          ← Open full periodic table and highlight element (e.g. Fe, Na, Au). Use for chemistry.
[TOOL_TIMELINE]date:event|date:event  ← Visual timeline. Separate events with |. Use for history, biology, physics history.
[TOOL_TABLE]H1,H2,H3\\nV1,V2,V3\\n...  ← Formatted data table. First line = headers. CSV format. Use for data comparison.
[TOOL_GEO]                     ← Open GeoGebra geometry workspace. Use for geometry, trigonometry, constructions.
[TOOL_VOCAB]word:pos:definition|word:pos:def  ← Vocabulary flashcards. Use for languages, terminology-heavy topics.
[TOOL_CLOSE]                   ← Close/hide the current tool when no longer needed.

## SPEECH
[SPEECH]One or two spoken sentences max. Will be read aloud — NO LaTeX, NO markdown.

## TEACHER RULES
1. **Speak then write**: always [SPEECH] first, then matching BOARD/TOOL content
2. **Tool discipline**: open a tool the moment it helps; close with [TOOL_CLOSE] when switching topics
   **CRITICAL — never place two TOOL_* markers back-to-back.** Always insert a [SPEECH] between them.
   Correct: `[TOOL_GRAPH]y=x^2` → `[SPEECH]Now let me compare this with a table.` → `[TOOL_CLOSE]` → `[TOOL_TABLE]...`
   Wrong:   `[TOOL_GRAPH]y=x^2` → `[TOOL_TABLE]...`  ← this closes the graph before the student can see it
3. **Subject adaptation — choose the MOST RELEVANT tool only**:
   - Math/Calculus/Physics → BOARD_FORMULA, TOOL_GRAPH, TOOL_EQUATION, TOOL_GEO
   - Chemistry → TOOL_PERIODIC, BOARD_FORMULA, TOOL_TABLE
   - History/Biology/Timeline topics → TOOL_TIMELINE, BOARD_TEXT, BOARD_HIGHLIGHT
   - Computer Science / Programming courses ONLY → TOOL_CODE
   - Economics/Statistics → TOOL_GRAPH, TOOL_TABLE, BOARD_FORMULA
   - Languages → TOOL_VOCAB, BOARD_TEXT, BOARD_HIGHLIGHT
4. **TOOL_CODE is for CS/programming students.** For math/science, prefer TOOL_GRAPH or TOOL_EQUATION. Only use TOOL_CODE when the student is explicitly studying programming or asks to see code.
5. **Canvas graphics work in TOOL_CODE.** The sandbox has `document.createElement('canvas')`, `document.body.appendChild()`, `requestAnimationFrame`, and `Math`. You can write animated canvas visualizations for CS students. Do NOT use this for explaining math to non-CS students — use TOOL_GRAPH instead.
6. **Keep speech short**: each [SPEECH] is 1–2 sentences for subtitle display
7. **No raw LaTeX in speech**: spell out formulas in words ("x squared plus 2x")
8. **Socratic**: after every 2–3 exchanges, end with a question that makes the student think
9. **Warm and precise**: correct errors gently, celebrate progress, build intuition
10. **If student draws**: describe what you see, identify what is correct and what needs fixing
11. **Multiple graphs**: call [TOOL_GRAPH] multiple times to overlay expressions on the same graph
12. **End with engagement**: always close your response with a [SPEECH] question or challenge
"""

        # Extract text part for history
        text_part = next((c["text"] for c in content if c.get("type") == "text"), "")
        if text_part:
            self.session.add_message("user", text_part)

        full_text = ""
        class_messages = self.session.get_history_for_api()[:-1] + [{"role": "user", "content": content}]

        @_retry_api
        def _call_class():
            return self.client.messages.stream(
                model=MODEL,
                max_tokens=3000,
                system=system,
                messages=class_messages,
            )

        with _call_class() as stream:
            for text in stream.text_stream:
                full_text += text
                yield text

        self.session.add_message("assistant", full_text)

    # ── Post-lesson quiz offer (called after lesson section ends) ──────

    def quiz_offer_message(self) -> str:
        topic = self.session.get_current_topic_name()
        weak  = self.tracker.get_weakest_skills(1)
        weak_str = f" (focus on {weak[0][0].replace('_',' ')})" if weak else ""
        return (f"You have just covered the key ideas of **{topic}**. "
                f"Would you like a short quiz{weak_str} to lock this in? "
                f"Studies show testing right after learning doubles retention.")


# ── Helpers ───────────────────────────────────────────────────────────

_SKILL_KEYWORDS: dict[str, list[str]] = {
    "linear_equations":   ["linear equation", "system of equations", "simultaneous"],
    "quadratics":         ["quadratic", "parabola", "discriminant", "completing the square"],
    "polynomials":        ["polynomial", "long division", "synthetic division", "factor theorem"],
    "rational_functions": ["rational function", "asymptote", "rational expression"],
    "derivatives":        ["derivative", "differentiation", "chain rule", "product rule"],
    "integrals":          ["integral", "antiderivative", "riemann", "fundamental theorem"],
    "limits":             ["limit", "continuity", "l'hopital"],
    "trigonometry":       ["sine", "cosine", "tangent", "trig", "unit circle", "radian"],
    "trig_identities":    ["pythagorean identity", "double angle", "sum and difference"],
    "exponential_log":    ["exponential", "logarithm", "natural log", "ln"],
    "probability":        ["probability", "bayes", "permutation", "combination"],
    "descriptive_stats":  ["mean", "median", "variance", "standard deviation"],
    "distributions":      ["normal distribution", "binomial", "z-score"],
}


def _detect_skill(text: str) -> Optional[str]:
    t = text.lower()
    best, best_n = None, 0
    for sid, kws in _SKILL_KEYWORDS.items():
        # Word-boundary matching prevents "log" matching inside "logarithm" etc.
        n = sum(1 for kw in kws if re.search(r'\b' + re.escape(kw) + r'\b', t))
        if n > best_n:
            best_n, best = n, sid
    return best if best_n >= 2 else None


def _apply_grading(response: str, tracker: MasteryTracker) -> None:
    pattern = r'\{"skill_id"\s*:\s*"([^"]+)"\s*,\s*"quality"\s*:\s*(\d)\}'
    for m in re.finditer(pattern, response):
        tracker.record_answer(m.group(1), int(m.group(2)))
