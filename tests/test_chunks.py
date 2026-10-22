"""Checks on the committed data/chunks.json -- the index is built from this
file, so a bad re-ingest should fail here before it reaches the bot.
"""

import json

import pytest

from index import CHUNKS_FILE
from qa import DOC_LABELS

REQUIRED_FIELDS = {"text", "doc_type", "source_file", "page", "kind", "section"}


@pytest.fixture(scope="module")
def chunks():
    return json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))


def test_every_chunk_has_source_metadata(chunks):
    for c in chunks:
        assert REQUIRED_FIELDS <= c.keys(), c
        assert c["text"].strip()
        assert c["doc_type"] in DOC_LABELS
        assert c["page"] >= 1
        assert c["kind"] in {"table_row", "prose"}
        assert c["section"] in {"A", "B", None}


def test_all_four_documents_are_indexed(chunks):
    assert {c["doc_type"] for c in chunks} == set(DOC_LABELS)


def test_escape_of_water_limits_row_is_intact(chunks):
    """The burst-pipe trick question depends on this exact row surviving
    table extraction with all three tiers and Section A attached.
    """
    row = next(c for c in chunks if c["text"].startswith("Escape of Water (Clause A2.3)"))
    assert "Silver: RM25,000" in row["text"] and "Gold: RM75,000" in row["text"]
    assert row["doc_type"] == "benefits_schedule" and row["section"] == "A"


def test_gold_flood_row_is_present_for_the_trap_question(chunks):
    """The flood trap question relies on this row existing (and the model
    NOT trusting it on its own) -- if it disappears, the test is meaningless.
    """
    row = next(c for c in chunks if c["text"].startswith("Flood — Buildings"))
    assert "Gold: 50% of Buildings Sum Insured" in row["text"]
