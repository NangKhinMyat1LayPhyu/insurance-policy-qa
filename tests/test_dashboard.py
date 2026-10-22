"""The admin dashboard renders, with Streamlit's headless test runner, on an
empty log and on a log with answered and flagged questions.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import server.db as db

DASHBOARD = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


@pytest.fixture
def log(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_FILE", tmp_path / "conversations.db")
    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret")
    db.init_db()
    return db


def run(logged_in=True):
    app = AppTest.from_file(DASHBOARD, default_timeout=30)
    if logged_in:
        app.session_state.authed = True
    app.run()
    assert not app.exception, app.exception
    return app


def test_empty_log_shows_getting_started_message(log):
    app = run()
    assert "No conversations yet" in app.info[0].value


def test_kpis_and_needs_human_queue(log):
    log.log_message("s1", "Excess for Escape of Water, Silver?", "RM1,000", False, None, None,
                    category="Escape of Water", tier="Silver")
    log.log_message("s1", "Excess for Escape of Water, Silver?", "RM1,000", True, 1.0, "Excess for Escape of Water, Silver?",
                    category="Escape of Water", tier="Silver")
    log.log_message("s2", "Do you cover my car?", "The provided documents do not answer this.", False, None, None,
                    category="Other, General", flag_reason="Documents don't answer it")

    app = run()
    kpis = {m.label: m.value for m in app.metric}
    assert kpis == {"Questions": "3", "Conversations": "2", "Answered by the bot": "67%", "Waiting for a human": "1"}
    assert app.tabs[0].label == "🚩 Needs a human (1)"

    app.button(key="resolve-3").click().run()
    assert {m.label: m.value for m in app.metric}["Waiting for a human"] == "0"


def test_password_gate(log):
    app = run(logged_in=False)
    assert not app.metric
    app.text_input[0].input("wrong").run()
    assert not app.metric and "Wrong password." in app.error[0].value
    app.text_input[0].input("s3cret").run()
    assert not app.exception
    assert "No conversations yet" in app.info[0].value


def test_locked_without_a_password(log, monkeypatch):
    monkeypatch.delenv("ADMIN_PASSWORD")
    app = run(logged_in=False)
    assert not app.metric and not app.text_input
    assert "ADMIN_PASSWORD" in app.error[0].value


def test_html_in_a_conversation_id_is_escaped(log):
    log.log_message("<b>x</b>", "Talk to a person?", "Flagged.", False, None, None,
                    category="Other, General", flag_reason="Customer asked for a person")
    app = run()
    shown = " ".join(m.value for m in app.markdown)
    assert "&lt;b&gt;x" in shown and "<b>x" not in shown
