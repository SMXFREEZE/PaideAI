"""
Mastery Tracker
===============
Tracks skill mastery using the SM-2 spaced repetition algorithm.
Each skill has: mastery (0-1), ease_factor, interval, next_review date.

SM-2 Reference: Wozniak (1990) — SuperMemo 2 algorithm
  - After each quiz answer, compute quality score (0-5)
  - Update ease factor: EF' = EF + (0.1 - (5-q)*(0.08 + (5-q)*0.02))
  - Update interval based on quality and history
  - Schedule next review

Improvements over baseline:
  - Atomic writes (write-then-rename) prevent data corruption
  - Daily decay guard: forgetting curve runs at most once per calendar day
  - Skill ID validation before recording (prevents typo skill IDs)
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional


MASTERY_FILE = Path(__file__).parents[2] / "data" / "mastery.json"

# SM-2 defaults
DEFAULT_EF       = 2.5
MIN_EF           = 1.3
INITIAL_INTERVALS = [1, 6]  # days for first two reps


# ── Atomic JSON write ──────────────────────────────────────────────────────────

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


class MasteryTracker:
    def __init__(self, path: Path = MASTERY_FILE):
        self.path = path
        self.data: dict[str, Any] = self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "skills": {},
            "session_count": 0,
            "last_updated": None,
            "last_decay_date": None,   # NEW: prevents multiple decays per day
        }

    def save(self) -> None:
        self.data["last_updated"] = datetime.now().isoformat()
        _atomic_write_json(self.path, self.data)

    # ── Skill record management ───────────────────────────────────────────────

    def _get_skill(self, skill_id: str) -> dict:
        if skill_id not in self.data["skills"]:
            self.data["skills"][skill_id] = {
                "skill_id":    skill_id,
                "mastery":     0.0,
                "ease_factor": DEFAULT_EF,
                "interval_days": 1,
                "repetitions": 0,
                "next_review": date.today().isoformat(),
                "history": [],
            }
        return self.data["skills"][skill_id]

    def is_valid_skill(self, skill_id: str, valid_ids: Optional[list[str]] = None) -> bool:
        """Return True if skill_id is in the known skill list (or if no list provided)."""
        if valid_ids is None:
            return True
        return skill_id in valid_ids

    # ── SM-2 core ─────────────────────────────────────────────────────────────

    def record_answer(self, skill_id: str, quality: int,
                      valid_ids: Optional[list[str]] = None) -> dict:
        """
        Record a quiz answer and update SM-2 schedule.

        quality: 0-5 (0=complete blackout, 5=perfect)
        valid_ids: if provided, rejects unknown skill IDs (prevents typo bloat)

        Returns updated skill record.
        """
        quality = max(0, min(5, quality))

        # Guard against typo skill IDs
        if valid_ids and not self.is_valid_skill(skill_id, valid_ids):
            # Silently ignore unknown skill to avoid mastery table bloat
            return {"skill_id": skill_id, "mastery": 0.0, "error": "unknown_skill"}

        skill = self._get_skill(skill_id)

        # Update ease factor
        ef = skill["ease_factor"]
        ef = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        skill["ease_factor"] = round(max(MIN_EF, ef), 3)

        # Update interval
        n = skill["repetitions"]
        if quality < 3:
            skill["repetitions"]  = 0
            skill["interval_days"] = 1
        else:
            skill["repetitions"] = n + 1
            if n == 0:
                skill["interval_days"] = INITIAL_INTERVALS[0]
            elif n == 1:
                skill["interval_days"] = INITIAL_INTERVALS[1]
            else:
                skill["interval_days"] = round(skill["interval_days"] * skill["ease_factor"])

        # Update mastery (rolling weighted average)
        mastery_delta = (quality / 5.0 - skill["mastery"]) * 0.3
        skill["mastery"] = max(0.0, min(1.0, round(skill["mastery"] + mastery_delta, 3)))

        # Schedule next review
        skill["next_review"] = (
            date.today() + timedelta(days=skill["interval_days"])
        ).isoformat()

        # Append to history (keep last 50)
        skill["history"].append({
            "date":        date.today().isoformat(),
            "quality":     quality,
            "mastery_after": skill["mastery"],
        })
        skill["history"] = skill["history"][-50:]

        self.save()
        return skill

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_mastery(self, skill_id: str) -> float:
        return self._get_skill(skill_id)["mastery"]

    def get_due_skills(self) -> list[str]:
        today = date.today().isoformat()
        return [s for s, r in self.data["skills"].items() if r["next_review"] <= today]

    def get_all_mastery(self) -> dict[str, float]:
        return {s: r["mastery"] for s, r in self.data["skills"].items()}

    def get_weakest_skills(self, n: int = 5) -> list[tuple[str, float]]:
        attempted = {s: r["mastery"] for s, r in self.data["skills"].items()
                     if r["repetitions"] > 0}
        return sorted(attempted.items(), key=lambda x: x[1])[:n]

    def get_strongest_skills(self, n: int = 5) -> list[tuple[str, float]]:
        attempted = {s: r["mastery"] for s, r in self.data["skills"].items()
                     if r["repetitions"] > 0}
        return sorted(attempted.items(), key=lambda x: x[1], reverse=True)[:n]

    def get_summary_json(self) -> dict:
        """Compact mastery summary for Claude prompts (minimal tokens)."""
        all_m = self.get_all_mastery()
        weak  = self.get_weakest_skills(3)
        due   = self.get_due_skills()
        avg   = round(sum(all_m.values()) / max(len(all_m), 1), 2)
        return {
            "avg": avg,
            "total": len(all_m),
            "weak":  [{"s": s, "m": round(m, 2)} for s, m in weak],
            "due":   due[:5],
        }

    def bump_exposure(self, skill_id: str, small_gain: float = 0.05) -> None:
        """Lightly increase mastery when topic is discussed (not quizzed)."""
        skill = self._get_skill(skill_id)
        skill["mastery"] = min(1.0, round(skill["mastery"] + small_gain, 3))
        self.save()

    def quality_from_grade(self, pct: float) -> int:
        for threshold, q in [(95, 5), (80, 4), (60, 3), (40, 2), (20, 1)]:
            if pct >= threshold:
                return q
        return 0

    def decay_all(self, decay_rate: float = 0.003) -> None:
        """
        Forgetting curve decay — runs at most ONCE per calendar day.
        Server restarts no longer trigger multiple decays.
        """
        today_str = date.today().isoformat()

        # Guard: skip if already decayed today
        if self.data.get("last_decay_date") == today_str:
            return

        changed = False
        today = date.today()
        for rec in self.data["skills"].values():
            if rec["repetitions"] == 0:
                continue
            last_str = (
                rec["history"][-1]["date"] if rec["history"]
                else rec.get("next_review", today_str)
            )
            days_since = (today - date.fromisoformat(last_str[:10])).days
            if days_since >= 3:
                decay = decay_rate * math.log1p(days_since - 2)
                new_m = max(0.0, round(rec["mastery"] - decay, 3))
                if new_m != rec["mastery"]:
                    rec["mastery"] = new_m
                    changed = True

        self.data["last_decay_date"] = today_str
        if changed:
            self.save()
        else:
            # Still save to record last_decay_date even with no changes
            self.save()
