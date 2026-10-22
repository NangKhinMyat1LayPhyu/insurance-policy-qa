"""The HTTP API, with the Q&A pipeline replaced by a fake answer."""

import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors
from langchain_core.documents import Document

import server.db as db
from server import app as app_module


def hit(doc_type, page, section="A"):
    return (Document(page_content="...", metadata={"doc_type": doc_type, "page": page, "section": section}), "vector", 0.1)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_FILE", tmp_path / "conversations.db")
    fake_hits = [hit("policy_wording", 6), hit("policy_wording", 6), hit("benefits_schedule", 2)]
    info = {"hit": False, "category": "Flood", "tier": "Gold"}
    monkeypatch.setattr(app_module.qa, "answer", lambda q: ("No, flood is not covered.", fake_hits, info))
    monkeypatch.setattr(app_module.qa, "rewrite_with_history", lambda q, history: q)
    with TestClient(app_module.app) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_root_redirects_to_demo_page(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/static/demo-insurer-site.html"


def test_chat_returns_answer_with_deduplicated_sources(client):
    body = client.post("/api/chat", json={"session_id": "s1", "message": "Flood, Gold tier?"}).json()
    assert body["answer"] == "No, flood is not covered."
    assert body["cache_hit"] is False
    assert body["needs_human"] is False
    assert body["sources"] == [
        {"document": "Policy Wording", "page": 6, "section": "A"},
        {"document": "Benefits Schedule", "page": 2, "section": "A"},
    ]


def test_rate_limit_returns_clean_503(client, monkeypatch):
    def rate_limited(q):
        raise genai_errors.ClientError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})

    monkeypatch.setattr(app_module.qa, "answer", rate_limited)
    response = client.post("/api/chat", json={"session_id": "s1", "message": "Flood, Gold tier?"})
    assert response.status_code == 503
    assert "try again" in response.json()["detail"]


def test_chat_is_logged(client):
    client.post("/api/chat", json={"session_id": "s1", "message": "Flood, Gold tier?"})
    conn = db.get_connection()
    rows = conn.execute("SELECT session_id, question, answer, cache_hit, category, tier, needs_human FROM messages").fetchall()
    conn.close()
    assert [tuple(r) for r in rows] == [("s1", "Flood, Gold tier?", "No, flood is not covered.", 0, "Flood", "Gold", 0)]


def test_request_for_a_person_is_flagged_and_logged(client):
    body = client.post("/api/chat", json={"session_id": "s2", "message": "Can I talk to a real person?"}).json()
    assert body["needs_human"] is True
    conn = db.get_connection()
    row = conn.execute("SELECT needs_human, flag_reason, resolved FROM messages").fetchone()
    conn.close()
    assert tuple(row) == (1, "Customer asked for a person", 0)


def test_follow_up_is_rewritten_with_history_and_both_versions_logged(client, monkeypatch):
    seen = {"history": [], "answered": []}

    def fake_rewrite(question, history):
        seen["history"].append(history)
        return "Which claim form for a flood claim, Gold tier?" if history else question

    def fake_answer(question):
        seen["answered"].append(question)
        return "Form HC-4.", [], {"hit": False}

    monkeypatch.setattr(app_module.qa, "rewrite_with_history", fake_rewrite)
    monkeypatch.setattr(app_module.qa, "answer", fake_answer)

    client.post("/api/chat", json={"session_id": "s1", "message": "Flood, Gold tier?"})
    client.post("/api/chat", json={"session_id": "s1", "message": "Which form for that?"})
    client.post("/api/chat", json={"session_id": "s2", "message": "Is theft covered?"})

    assert seen["history"] == [[], [("Flood, Gold tier?", "Form HC-4.")], []]
    assert seen["answered"] == ["Flood, Gold tier?", "Which claim form for a flood claim, Gold tier?", "Is theft covered?"]

    conn = db.get_connection()
    rows = conn.execute("SELECT question, standalone_question FROM messages ORDER BY id").fetchall()
    conn.close()
    assert [tuple(r) for r in rows] == [
        ("Flood, Gold tier?", None),
        ("Which form for that?", "Which claim form for a flood claim, Gold tier?"),
        ("Is theft covered?", None),
    ]
