"""PDF chunking rules that the answers depend on: Section A/B tagging, and
table rows keeping their tier column headers.
"""

from ingest import clean_prose, detect_section, table_rows_as_chunks


def test_detects_section_headings():
    assert detect_section("Part 3 — Section A: HomeCare\nWhat is covered", None) == "A"
    assert detect_section("Part 4 — Section B: Health\nInpatient benefits", "A") == "B"


def test_section_carries_forward_on_pages_without_a_heading():
    assert detect_section("A2.3 Escape of Water means ...", "A") == "A"


def test_clause_references_in_body_text_do_not_change_section():
    assert detect_section("See Clause A1.1 and Exclusion 5.2.2 for details.", "B") == "B"


def test_general_conditions_reset_to_general():
    assert detect_section("Part 6 — General Conditions", "B") is None


def test_last_heading_on_a_page_wins():
    page = "5.2 Exclusions applicable to Section A\n...\n5.3 Exclusions applicable to Section B"
    assert detect_section(page, None) == "B"


def test_strips_running_header_and_footer():
    text = (
        "Lumina Kasturi Assurance Berhad — SYNTHETIC — FOR TESTING ONLY\n"
        "A2.3 Escape of Water\n"
        "LKA/HHP/PW/2026-01 Page 4 of 15"
    )
    assert clean_prose(text) == "A2.3 Escape of Water"


class FakeTable:
    def __init__(self, rows):
        self.rows = rows

    def extract(self):
        return self.rows


def make_chunks(rows):
    return table_rows_as_chunks(FakeTable(rows), "benefits_schedule", "03_Benefits_Schedule.pdf", 2, "A")


def test_each_row_carries_its_tier_headers():
    rows = [
        ["Benefit", "Silver", "Gold", "Platinum"],
        ["Escape of Water\n(Clause A2.3)", "RM25,000", "RM75,000", "Up to Sum Insured"],
    ]
    [chunk] = make_chunks(rows)
    assert chunk["text"] == "Escape of Water (Clause A2.3) -- Silver: RM25,000; Gold: RM75,000; Platinum: Up to Sum Insured."
    assert chunk["kind"] == "table_row" and chunk["section"] == "A" and chunk["page"] == 2


def test_skips_empty_and_unlabelled_rows():
    rows = [
        ["Benefit", "Silver", "Gold", "Platinum"],
        [None, None, None, None],
        [None, "RM1", "RM2", "RM3"],
        ["Theft", None, None, None],
    ]
    assert make_chunks(rows) == []


def test_header_only_table_gives_no_chunks():
    assert make_chunks([["Benefit", "Silver", "Gold", "Platinum"]]) == []
