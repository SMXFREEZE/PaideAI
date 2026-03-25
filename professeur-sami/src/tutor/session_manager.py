"""
Session Manager
===============
Manages session state with full profile persistence to data/profile.json.
Profile survives server restarts. Includes study plan + daily tracking.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically: write to temp file then rename, preventing corruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        Path(tmp).replace(path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


CURRICULUM_DIR = Path(__file__).parents[2] / "data" / "curricula"
DEFAULT_CURRICULUM = CURRICULUM_DIR / "default.yaml"
PROFILE_FILE = Path(__file__).parents[2] / "data" / "profile.json"

MICRO_QUIZ_INTERVAL = int(os.environ.get("MICRO_QUIZ_INTERVAL_MINUTES", "15")) * 60
MAX_HISTORY = int(os.environ.get("MAX_CONVERSATION_HISTORY", "40"))


@dataclass
class Message:
    role: str
    content: str


@dataclass
class SessionState:
    # Identity
    student_name: str = ""
    interests: list = field(default_factory=list)

    # Course
    course_name: str = ""
    course_outline: str = ""      # raw syllabus/notes text
    exam_weeks: int = 8
    daily_minutes: int = 30
    study_start_date: str = ""
    study_plan: list = field(default_factory=list)
    onboarding_done: bool = False

    # Session navigation
    current_topic: Optional[str] = None
    current_unit: Optional[str] = None
    history: list = field(default_factory=list)
    session_start: float = field(default_factory=time.time)
    last_quiz_time: float = field(default_factory=time.time)
    quiz_count: int = 0
    curriculum: dict = field(default_factory=dict)

    # Progress tracking
    daily_log: list = field(default_factory=list)
    streak_days: int = 0


class SessionManager:
    def __init__(self):
        self.state = SessionState(student_name=os.environ.get("STUDENT_NAME", ""))
        self._load_profile()
        self._load_curriculum()

    # ------------------------------------------------------------------
    # Profile persistence
    # ------------------------------------------------------------------

    def _load_curriculum(self) -> None:
        if DEFAULT_CURRICULUM.exists():
            with open(DEFAULT_CURRICULUM, "r", encoding="utf-8") as f:
                self.state.curriculum = yaml.safe_load(f)
        else:
            self.state.curriculum = {"name": "General Mathematics", "units": []}

    def _load_profile(self) -> None:
        if not PROFILE_FILE.exists():
            return
        try:
            with open(PROFILE_FILE, "r", encoding="utf-8") as f:
                p = json.load(f)
            for key in [
                "student_name", "interests", "course_name", "course_outline",
                "exam_weeks", "daily_minutes", "study_start_date", "study_plan",
                "onboarding_done", "daily_log", "streak_days",
            ]:
                if key in p:
                    setattr(self.state, key, p[key])
        except Exception:
            pass

    def save_profile(self) -> None:
        data = {
            "student_name": self.state.student_name,
            "interests": self.state.interests,
            "course_name": self.state.course_name,
            "course_outline": self.state.course_outline,
            "exam_weeks": self.state.exam_weeks,
            "daily_minutes": self.state.daily_minutes,
            "study_start_date": self.state.study_start_date,
            "study_plan": self.state.study_plan,
            "onboarding_done": self.state.onboarding_done,
            "daily_log": self.state.daily_log,
            "streak_days": self.state.streak_days,
        }
        _atomic_write_json(PROFILE_FILE, data)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        self.state.history.append(Message(role=role, content=content))
        if len(self.state.history) > MAX_HISTORY:
            self.state.history = self.state.history[:2] + self.state.history[-(MAX_HISTORY - 2):]

    def get_history_for_api(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.state.history]

    def clear_history(self) -> None:
        self.state.history = []

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def check_micro_quiz_due(self) -> bool:
        return (time.time() - self.state.last_quiz_time) >= MICRO_QUIZ_INTERVAL

    def mark_quiz_done(self) -> None:
        self.state.last_quiz_time = time.time()
        self.state.quiz_count += 1

    def seconds_until_next_quiz(self) -> int:
        return max(0, int(MICRO_QUIZ_INTERVAL - (time.time() - self.state.last_quiz_time)))

    def elapsed_minutes(self) -> float:
        return (time.time() - self.state.session_start) / 60

    # ------------------------------------------------------------------
    # Topic
    # ------------------------------------------------------------------

    def set_topic(self, topic_id: str, unit_id: Optional[str] = None) -> None:
        self.state.current_topic = topic_id
        if unit_id:
            self.state.current_unit = unit_id

    def get_current_topic_name(self) -> str:
        if self.state.current_topic:
            for unit in self.state.curriculum.get("units", []):
                for topic in unit.get("topics", []):
                    if topic["id"] == self.state.current_topic:
                        return topic["name"]
            return self.state.current_topic
        return self.state.course_name or "Mathematics"

    # ------------------------------------------------------------------
    # Curriculum / outline text for prompts
    # ------------------------------------------------------------------

    def get_curriculum_text(self) -> str:
        if self.state.course_outline:
            return self.state.course_outline[:4000]
        c = self.state.curriculum
        lines = [f"# {c.get('name', 'Mathematics')}"]
        for unit in c.get("units", []):
            lines.append(f"\n## {unit['name']}")
            for topic in unit.get("topics", []):
                lines.append(f"  - {topic['name']}")
        return "\n".join(lines)

    def get_all_skill_ids(self) -> list[str]:
        ids = []
        for unit in self.state.curriculum.get("units", []):
            ids.append(unit["id"])
            for topic in unit.get("topics", []):
                ids.append(topic["id"])
        return ids

    def set_interests(self, interests: list[str]) -> None:
        self.state.interests = interests

    def get_interests_str(self) -> str:
        return ", ".join(self.state.interests) if self.state.interests else "technology, problem-solving"

    # ------------------------------------------------------------------
    # Study plan helpers
    # ------------------------------------------------------------------

    def get_current_week(self) -> int:
        if not self.state.study_start_date:
            return 1
        try:
            start = date.fromisoformat(self.state.study_start_date)
            delta = (date.today() - start).days
            return min(max(1, delta // 7 + 1), self.state.exam_weeks)
        except Exception:
            return 1

    def get_today_topic(self) -> Optional[dict]:
        week = self.get_current_week()
        plan = self.state.study_plan
        if not plan:
            return None
        week_data = next((w for w in plan if w.get("week") == week), None)
        if not week_data:
            week_data = plan[0] if plan else None
        if not week_data:
            return None
        day_name = date.today().strftime("%a")
        for d in week_data.get("daily", []):
            if d.get("day") == day_name:
                return {"week": week, **d}
        topics = week_data.get("topics", [])
        return {"week": week, "topic": topics[0] if topics else "Review", "duration": self.state.daily_minutes, "type": "lesson"}

    def log_session(self, topic: str, duration_min: int, quiz_score: Optional[float] = None) -> None:
        entry = {
            "date": date.today().isoformat(),
            "topic": topic,
            "duration_min": duration_min,
            "quiz_score": quiz_score,
        }
        self.state.daily_log.append(entry)
        self._update_streak()
        self.save_profile()

    def _update_streak(self) -> None:
        dates = sorted({e["date"] for e in self.state.daily_log}, reverse=True)
        if not dates:
            return
        streak: int = 0
        check: date = date.today()
        for d in dates:
            d_date = date.fromisoformat(d)
            if d_date == check:
                streak = streak + 1
                check = check - timedelta(days=1)
            elif d_date < check:
                break
        self.state.streak_days = streak
