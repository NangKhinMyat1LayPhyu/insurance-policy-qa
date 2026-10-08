import os
import sqlite3
import time
from pathlib import Path

DEFAULT_DB_FILE = Path(__file__).resolve().parents[1] / "data" / "conversations.db"
DB_FILE = Path(os.getenv("CONVERSATIONS_DB", DEFAULT_DB_FILE))

ADDED_COLUMNS = {
    "category": "TEXT",
    "tier": "TEXT",
    "needs_human": "INTEGER NOT NULL DEFAULT 0",
    "flag_reason": "TEXT",
    "resolved": "INTEGER NOT NULL DEFAULT 0",
}


def get_connection():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            time TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            cache_hit INTEGER NOT NULL,
            similarity REAL,
            matched_question TEXT
        )"""
    )
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
    for name, definition in ADDED_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE messages ADD COLUMN {name} {definition}")
    conn.commit()
    conn.close()


def log_message(session_id, question, answer, cache_hit, similarity, matched_question,
                category=None, tier=None, flag_reason=None):
    conn = get_connection()
    conn.execute(
        """INSERT INTO messages
           (session_id, time, question, answer, cache_hit, similarity, matched_question,
            category, tier, needs_human, flag_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, time.strftime("%Y-%m-%d %H:%M:%S"), question, answer, int(cache_hit), similarity,
         matched_question, category, tier, int(flag_reason is not None), flag_reason),
    )
    conn.commit()
    conn.close()


def set_resolved(message_id, resolved=True):
    conn = get_connection()
    conn.execute("UPDATE messages SET resolved = ? WHERE id = ?", (int(resolved), message_id))
    conn.commit()
    conn.close()
