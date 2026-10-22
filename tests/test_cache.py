"""The semantic cache's safety rules. The point of the category gate is that
a near-identical question about a different peril or tier must NEVER get a
cached answer, however similar the embeddings are.
"""

import pytest

import cache
from cache import category_match, cosine_sim, find_best_match

FLOOD_GOLD = {"category": "Flood", "tier": "Gold"}


def test_same_category_and_tier_matches():
    assert category_match(FLOOD_GOLD, {"category": "Flood", "tier": "Gold"})


def test_different_category_never_matches():
    assert not category_match(FLOOD_GOLD, {"category": "Theft", "tier": "Gold"})


def test_different_stated_tiers_never_match():
    assert not category_match(FLOOD_GOLD, {"category": "Flood", "tier": "Silver"})


def test_tier_stated_on_one_side_only_still_matches():
    assert category_match({"category": "Waiting Period", "tier": None}, {"category": "Waiting Period", "tier": "Gold"})


@pytest.mark.parametrize("other", [{"category": None, "tier": "Gold"}, {}])
def test_unknown_category_fails_closed(other):
    assert not category_match(FLOOD_GOLD, other)
    assert not category_match(other, FLOOD_GOLD)
    assert not category_match(other, other)


def entry(question, embedding, category, tier=None, version="v1"):
    return {"question": question, "embedding": embedding, "category": category, "tier": tier,
            "answer": f"answer to {question}", "doc_version": version}


def test_skips_entries_from_old_document_version():
    entries = [entry("old", [1, 0], "Flood", version="v0")]
    assert find_best_match([1, 0], {"category": "Flood"}, entries, "v1") == (None, 0.0)


def test_identical_embedding_in_wrong_category_is_ignored():
    entries = [entry("excess for theft", [1, 0], "Theft"), entry("flood cover", [0.6, 0.8], "Flood")]
    best, sim = find_best_match([1, 0], {"category": "Flood"}, entries, "v1")
    assert best["question"] == "flood cover"
    assert sim == pytest.approx(0.6)


def test_picks_most_similar_eligible_entry():
    entries = [entry("a", [0, 1], "Flood"), entry("b", [1, 0.1], "Flood")]
    best, _ = find_best_match([1, 0], {"category": "Flood"}, entries, "v1")
    assert best["question"] == "b"


def test_cosine_sim_handles_zero_vector():
    assert cosine_sim([0, 0], [1, 0]) == 0.0


class FakeEmbedder:
    """Every question embeds to the same vector -- the worst case for the
    cache, where similarity alone would say every pair is a match.
    """

    def embed_query(self, text):
        return [1.0, 0.0]


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    chunks = tmp_path / "chunks.json"
    chunks.write_text("[]")
    monkeypatch.setattr(cache, "CACHE_FILE", tmp_path / "cache.json")
    monkeypatch.setattr(cache, "CACHE_LOG_FILE", tmp_path / "cache_log.jsonl")
    monkeypatch.setattr(cache, "active_chunks_file", lambda: chunks)
    monkeypatch.setattr(cache, "get_embedder", lambda: FakeEmbedder())
    return chunks


def test_lookup_hits_same_category_and_tier(isolated_cache):
    cache.store("Excess for Escape of Water, Silver?", "RM1,000", question_terms={"category": "Escape of Water", "tier": "Silver"})
    result = cache.lookup("What's the Silver excess for escape of water?", question_terms={"category": "Escape of Water", "tier": "Silver"})
    assert result["hit"] and result["answer"] == "RM1,000"


def test_lookup_blocks_near_miss_despite_identical_embedding(isolated_cache):
    cache.store("Excess for Escape of Water, Silver?", "RM1,000", question_terms={"category": "Escape of Water", "tier": "Silver"})
    result = cache.lookup("Excess for Theft, Silver?", question_terms={"category": "Theft", "tier": "Silver"})
    assert not result["hit"] and result["answer"] is None


def test_lookup_misses_when_classification_failed(isolated_cache):
    cache.store("Excess for Escape of Water, Silver?", "RM1,000", question_terms={"category": "Escape of Water", "tier": "Silver"})
    assert not cache.lookup("Excess for Escape of Water, Silver?", question_terms={"category": None, "tier": None})["hit"]


def test_changing_documents_invalidates_cache(isolated_cache):
    terms = {"category": "Flood", "tier": "Gold"}
    cache.store("Is flood covered on Gold?", "Only with the Special Perils Extension.", question_terms=terms)
    isolated_cache.write_text('[{"text": "updated policy"}]')
    assert not cache.lookup("Is flood covered on Gold?", question_terms=terms)["hit"]
