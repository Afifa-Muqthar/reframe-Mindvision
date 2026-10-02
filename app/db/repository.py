"""Read/write helpers for the sessions table."""

import json
from datetime import datetime, timezone

from .schema import get_connection


def save_session(case_frame, narrative: str, image_path: str = None, audio_path: str = None) -> int:
    conn = get_connection()
    cur = conn.execute(
        """
        INSERT INTO sessions
            (input_text, emotion_scores, core_emotion, distortions, techniques,
             narrative, image_path, audio_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            case_frame.raw_text,
            json.dumps(case_frame.emotion_scores),
            case_frame.core_emotion,
            json.dumps(case_frame.distortions),
            json.dumps(case_frame.selected_techniques),
            narrative,
            image_path,
            audio_path,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    session_id = cur.lastrowid
    conn.close()
    return session_id


def update_image_path(session_id: int, image_path: str):
    conn = get_connection()
    conn.execute("UPDATE sessions SET image_path = ? WHERE id = ?", (image_path, session_id))
    conn.commit()
    conn.close()


def update_audio_path(session_id: int, audio_path: str):
    conn = get_connection()
    conn.execute("UPDATE sessions SET audio_path = ? WHERE id = ?", (audio_path, session_id))
    conn.commit()
    conn.close()


def get_session(session_id: int):
    conn = get_connection()
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_sessions(limit: int = 50):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def find_recurring_themes(min_occurrences: int = 3):
    """
    Very simple recurring-theme signal (Phase 2): counts how often each
    distortion has appeared across past sessions. Distortion type is used
    as the "theme" proxy (rather than raw topic extraction, which is out of
    scope) since it captures a recurring pattern of thinking, not just a
    recurring subject. If a distortion has appeared >= min_occurrences
    times, it's surfaced so it can be folded into future narrative prompts.
    """
    conn = get_connection()
    rows = conn.execute("SELECT distortions FROM sessions").fetchall()
    conn.close()

    counts = {}
    for row in rows:
        distortions = json.loads(row["distortions"])
        for d in distortions:
            counts[d] = counts.get(d, 0) + 1

    return {d: c for d, c in counts.items() if c >= min_occurrences}


def build_recurring_theme_note(min_occurrences: int = 3) -> str:
    """
    Human-readable version of find_recurring_themes, e.g. "self-labeling
    has come up 3 times recently". Returns "" if nothing recurs yet.
    """
    themes = find_recurring_themes(min_occurrences)
    if not themes:
        return ""
    top_theme, count = max(themes.items(), key=lambda kv: kv[1])
    readable = top_theme.replace("_", " ")
    return f"{readable} has come up {count} times in recent sessions"
