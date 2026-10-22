"""The conversation log, including upgrading a database created before the
dashboard columns existed.
"""

import sqlite3

import pytest

import server.db as db


@pytest.fixture
def db_file(tmp_path, monkeypatch):
    path = tmp_path / "conversations.db"
    monkeypatch.setattr(db, "DB_FILE", path)
    return path


def columns(path):
    conn = sqlite3.connect(path)
    names = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
    conn.close()
    return names


def test_old_database_is_upgraded_without_losing_rows(db_file):
    conn = sqlite3.connect(db_file)
    conn.execute("""CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                    time TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL, cache_hit INTEGER NOT NULL,
                    similarity REAL, matched_question TEXT)""")
    conn.execute("INSERT INTO messages (session_id, time, question, answer, cache_hit) VALUES ('s', 't', 'q', 'a', 0)")
    conn.commit()
    conn.close()

    db.init_db()
    db.init_db()

    assert set(db.ADDED_COLUMNS) <= columns(db_file)
    conn = db.get_connection()
    row = conn.execute("SELECT question, needs_human, resolved FROM messages").fetchone()
    conn.close()
    assert tuple(row) == ("q", 0, 0)


def test_flagged_message_can_be_resolved_and_reopened(db_file):
    db.init_db()
    db.log_message("s1", "Talk to a person", "...", False, None, None,
                   category="Other, General", tier=None, flag_reason="Customer asked for a person")

    def state():
        conn = db.get_connection()
        row = conn.execute("SELECT id, needs_human, flag_reason, resolved, category FROM messages").fetchone()
        conn.close()
        return row

    row = state()
    assert (row["needs_human"], row["flag_reason"], row["resolved"], row["category"]) == (1, "Customer asked for a person", 0, "Other, General")
    db.set_resolved(row["id"])
    assert state()["resolved"] == 1
    db.set_resolved(row["id"], False)
    assert state()["resolved"] == 0


def test_recent_turns_are_per_conversation_oldest_first_and_limited(db_file):
    db.init_db()
    for i in range(4):
        db.log_message("s1", f"q{i}", f"a{i}", False, None, None)
    db.log_message("other", "not mine", "x", False, None, None)
    assert db.recent_turns("s1", 3) == [("q1", "a1"), ("q2", "a2"), ("q3", "a3")]
    assert db.recent_turns("nobody", 3) == []


def test_recent_turns_prefer_the_rewritten_question(db_file):
    db.init_db()
    db.log_message("s1", "Burst pipe, Silver?", "Covered.", False, None, None)
    db.log_message("s1", "Which form for that?", "HC-2.", False, None, None,
                   standalone_question="Which claim form for a burst pipe (Escape of Water), Silver tier?")
    assert db.recent_turns("s1", 3)[-1][0] == "Which claim form for a burst pipe (Escape of Water), Silver tier?"
