"""Parsing the classifier's reply. The cache gate depends on this returning
None for anything it can't map to the closed category list -- an invented
category must never be treated as a real one.
"""

from terms import CATEGORIES, _parse, search_terms


def test_parses_exact_category_and_tier():
    assert _parse("category: Escape of Water\ntier: Silver") == {"category": "Escape of Water", "tier": "Silver"}


def test_parse_is_case_insensitive_and_returns_canonical_spelling():
    assert _parse("CATEGORY: escape of water\nTier: GOLD") == {"category": "Escape of Water", "tier": "Gold"}


def test_category_not_on_the_list_is_unknown():
    assert _parse("category: Burst Pipe\ntier: Silver")["category"] is None


def test_tier_none_is_unknown():
    assert _parse("category: Flood\ntier: none")["tier"] is None


def test_garbage_reply_is_fully_unknown():
    assert _parse("Sorry, I can't help with that.") == {"category": None, "tier": None}


def test_categories_are_unique():
    assert len({c.lower() for c in CATEGORIES}) == len(CATEGORIES)


def test_search_terms_skips_unknowns():
    assert search_terms({"category": "Flood", "tier": "Gold"}) == "Flood, Gold"
    assert search_terms({"category": "Flood", "tier": None}) == "Flood"
    assert search_terms({"category": None, "tier": None}) == ""
