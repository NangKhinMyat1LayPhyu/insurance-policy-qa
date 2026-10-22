"""answer()'s handling of the model's reply, with retrieval, classification
and Gemini all faked.
"""

from types import SimpleNamespace

import pytest
from google.genai import types

import qa


class FakeClient:
    def __init__(self, text, finish_reason):
        response = SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason=finish_reason)])
        self.models = SimpleNamespace(generate_content=lambda **kwargs: response)


@pytest.fixture
def stored(monkeypatch):
    stored = []
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(qa, "classify_question", lambda q: {"category": "Escape of Water", "tier": "Silver"})
    monkeypatch.setattr(qa, "retrieve", lambda q, question_terms=None, release=None: [])
    monkeypatch.setattr(qa.cache, "lookup", lambda *a, **k: {"hit": False})
    monkeypatch.setattr(qa.cache, "store", lambda q, text, question_terms=None: stored.append(text))
    return stored


def use_reply(monkeypatch, text, finish_reason):
    monkeypatch.setattr(qa.genai, "Client", lambda: FakeClient(text, finish_reason))


def test_complete_answer_is_returned_and_cached(monkeypatch, stored):
    use_reply(monkeypatch, "Covered up to RM25,000 (Source: Benefits Schedule, page 2).", types.FinishReason.STOP)
    text, _, info = qa.answer("burst pipe, Silver?")
    assert text.startswith("Covered up to RM25,000")
    assert stored == [text]
    assert not info.get("incomplete")


def test_truncated_answer_is_never_shown_or_cached(monkeypatch, stored):
    use_reply(monkeypatch, "Yes, covered, provided the damage did not continue for more than", types.FinishReason.MAX_TOKENS)
    text, _, info = qa.answer("burst pipe, Silver?")
    assert text == qa.INCOMPLETE_ANSWER
    assert info["incomplete"] is True
    assert stored == []


def test_empty_answer_is_treated_as_incomplete(monkeypatch, stored):
    use_reply(monkeypatch, None, types.FinishReason.STOP)
    assert qa.answer("burst pipe, Silver?")[0] == qa.INCOMPLETE_ANSWER
    assert stored == []


class FakeStore:
    def __init__(self):
        self.queries = []

    def similarity_search_with_score(self, query, k):
        self.queries.append(query)
        return [(SimpleNamespace(page_content=f"vector:{query}:{i}"), 0.1) for i in range(k)]


class FakeBM25:
    def __init__(self):
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return [SimpleNamespace(page_content="shared")] + [SimpleNamespace(page_content=f"kw:{query}:{i}") for i in range(9)]


@pytest.fixture
def fake_search(monkeypatch):
    store, bm25 = FakeStore(), FakeBM25()
    monkeypatch.setattr(qa, "load_store", lambda index_dir: store)
    monkeypatch.setattr(qa, "load_bm25", lambda chunks_file: bm25)
    return store, bm25


def test_terms_only_pass_searches_the_canonical_terms_alone(fake_search):
    store, bm25 = fake_search
    hits = qa.retrieve("how long before I can claim for hospitalisation?",
                       question_terms={"category": "Waiting Period", "tier": None})
    assert bm25.queries == ["how long before I can claim for hospitalisation? Waiting Period", "Waiting Period"]
    assert store.queries[-1] == qa.QUERY_PREFIX + "Waiting Period"
    texts = [doc.page_content for doc, _m, _s in hits]
    assert len(texts) == len(set(texts))
    assert len(texts) == qa.VECTOR_K + qa.TERMS_K + 1 + (qa.BM25_K - 1) + (qa.TERMS_K - 1)


def test_no_terms_pass_when_classification_failed(fake_search):
    store, bm25 = fake_search
    qa.retrieve("something odd", question_terms={"category": None, "tier": None})
    assert len(store.queries) == 1 and len(bm25.queries) == 1


class RecordingClient:
    def __init__(self, text, finish_reason=types.FinishReason.STOP):
        self.calls = []
        response = SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason=finish_reason)])

        def generate_content(**kwargs):
            self.calls.append(kwargs)
            return response

        self.models = SimpleNamespace(generate_content=generate_content)


HISTORY = [("My kitchen pipe burst. Silver tier - am I covered?", "Yes, as Escape of Water, up to RM25,000.")]


def use_client(monkeypatch, client):
    monkeypatch.setattr(qa.genai, "Client", lambda: client)
    return client


def test_first_message_is_not_rewritten_and_costs_no_call(monkeypatch):
    client = use_client(monkeypatch, RecordingClient("should not be used"))
    assert qa.rewrite_with_history("Is flood covered?", []) == "Is flood covered?"
    assert client.calls == []


def test_follow_up_is_rewritten_using_the_conversation(monkeypatch):
    client = use_client(monkeypatch, RecordingClient('"Which claim form do I use for an Escape of Water claim, Silver tier?"\n'))
    result = qa.rewrite_with_history("Which form do I use for that?", HISTORY)
    assert result == "Which claim form do I use for an Escape of Water claim, Silver tier?"
    sent = client.calls[0]["contents"]
    assert "My kitchen pipe burst. Silver tier" in sent and "Latest message: Which form do I use for that?" in sent


def test_long_earlier_answers_are_trimmed(monkeypatch):
    client = use_client(monkeypatch, RecordingClient("standalone"))
    qa.rewrite_with_history("and that?", [("q", "x" * 5000)])
    assert "x" * qa.HISTORY_ANSWER_CHARS in client.calls[0]["contents"]
    assert "x" * (qa.HISTORY_ANSWER_CHARS + 1) not in client.calls[0]["contents"]


@pytest.mark.parametrize("text, finish_reason", [("Which claim fo", types.FinishReason.MAX_TOKENS), ("", types.FinishReason.STOP), (None, types.FinishReason.STOP)])
def test_failed_rewrite_falls_back_to_the_original_question(monkeypatch, text, finish_reason):
    use_client(monkeypatch, RecordingClient(text, finish_reason))
    assert qa.rewrite_with_history("Which form for that?", HISTORY) == "Which form for that?"


def test_staged_release_answers_never_touch_the_cache(monkeypatch, stored):
    def no_cache(*args, **kwargs):
        raise AssertionError("cache used for a staged release")

    monkeypatch.setattr(qa.cache, "lookup", no_cache)
    monkeypatch.setattr(qa.cache, "store", no_cache)
    seen = {}

    def fake_retrieve(q, question_terms=None, release=None):
        seen["release"] = release
        return []

    monkeypatch.setattr(qa, "retrieve", fake_retrieve)
    use_reply(monkeypatch, "New wording: 90 days.", types.FinishReason.STOP)
    assert qa.answer("Waiting period for a Specified Illness?", release="/lib/staging/r2")[0] == "New wording: 90 days."
    assert seen["release"] == "/lib/staging/r2"
