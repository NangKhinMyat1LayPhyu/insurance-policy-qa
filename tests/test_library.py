"""The document library: stage a replacement, review the check report,
publish, roll back. PDF parsing and the search index are faked -- each fake
"PDF" is a text file with one passage per line, `section|kind|text` -- so
these tests exercise the release logic, not pdfplumber or FAISS.
"""

import json
from datetime import date, timedelta

import pytest

import cache
import ingest
import library

LIVE_POLICY = """None|prose|Policy Wording. Part 1 General.
A|prose|Escape of Water continuing more than 14 days is excluded.
A|table_row|Escape of Water -- Silver: RM25,000.
B|prose|The Waiting Period for Specified Illnesses is 120 days.
B|prose|The Waiting Period for Illness is 30 days."""

V2_POLICY = """None|prose|Policy Wording. Version 2.
A|prose|Escape of Water continuing more than 7 days is excluded.
A|table_row|Escape of Water -- Silver: RM25,000.
B|prose|The Waiting Period for Specified Illnesses is 90 days.
B|prose|The Waiting Period for Illness is 30 days."""


def fake_text(path):
    return path.read_text(encoding="utf-8")


def fake_build_chunks(doc_paths):
    chunks = []
    for doc in ingest.DOCUMENTS:
        path = doc_paths[doc["doc_type"]]
        text = fake_text(path)
        if text.startswith("BROKEN"):
            raise ValueError("not a PDF")
        for i, line in enumerate(text.splitlines(), start=1):
            section, kind, passage = line.split("|", 2)
            chunks.append({"text": passage, "doc_type": doc["doc_type"], "source_file": path.name, "page": i,
                           "kind": kind, "section": None if section == "None" else section})
    return chunks


@pytest.fixture
def lib(tmp_path, monkeypatch):
    originals = tmp_path / "repo"
    originals.mkdir()
    for doc in ingest.DOCUMENTS:
        body = LIVE_POLICY if doc["doc_type"] == "policy_wording" else f"None|prose|{doc['label']}. Content."
        (originals / doc["file"]).write_text(body, encoding="utf-8")

    monkeypatch.setenv("LIBRARY_DIR", str(tmp_path / "library"))
    monkeypatch.setattr(ingest, "ROOT", originals)
    monkeypatch.setattr(library, "CHUNKS_FILE", tmp_path / "no-chunks.json")
    monkeypatch.setattr(library, "INDEX_DIR", tmp_path / "no-index")
    monkeypatch.setattr(library.ingest, "build_chunks", fake_build_chunks)
    monkeypatch.setattr(library, "build_index", lambda chunks, out: (out.mkdir(parents=True), (out / "index.faiss").write_text("x")))
    monkeypatch.setattr(library, "_page_count", lambda path: len(fake_text(path).splitlines()))
    monkeypatch.setattr(library, "_sentences", lambda path: [line.split("|", 2)[2] for line in fake_text(path).splitlines()])
    library.ensure_initialized()
    return library


def stage(lib, body=V2_POLICY, effective=None, doc_type="policy_wording"):
    return lib.stage_replacement(doc_type, body.encode(), "02 Policy Wording v2.pdf", "v2", effective, "test")


def test_without_library_dir_the_repo_data_is_live(monkeypatch):
    library.ensure_initialized()
    assert library.active_release_id() is None
    assert library.active_paths() == (library.CHUNKS_FILE, library.INDEX_DIR)


def test_first_start_seeds_an_initial_release(lib):
    assert lib.active_release_id() == "initial"
    lib.ensure_initialized()
    chunks_file, index_dir = lib.active_paths()
    assert (index_dir / "index.faiss").exists()
    assert len(json.loads(chunks_file.read_text())) == 5 + 3
    assert [e["action"] for e in lib.history()] == ["initialise"]


def test_staged_release_is_not_live(lib):
    staged = stage(lib)
    assert lib.active_release_id() == "initial"
    manifest = lib.read_manifest(lib.release_dir(staged, staged=True))
    assert manifest["based_on"] == "initial" and manifest["changed"] == "policy_wording" and manifest["published"] is None
    assert manifest["documents"]["policy_wording"]["file"] == "02_Policy_Wording_v2.pdf"
    assert manifest["documents"]["benefits_schedule"]["version"] == "initial"


def test_check_report_shows_what_changed(lib):
    report = lib.check_report(stage(lib))
    assert report["errors"] == [] and report["warnings"] == []
    assert "The Waiting Period for Specified Illnesses is 90 days." in report["added"]
    assert "The Waiting Period for Specified Illnesses is 120 days." in report["removed"]
    assert "The Waiting Period for Illness is 30 days." not in report["added"] + report["removed"]
    assert (report["old"]["version"], report["new"]["version"]) == ("initial", "v2")


def test_identical_file_is_an_error_and_cannot_be_published(lib):
    staged = stage(lib, body=LIVE_POLICY)
    assert lib.check_report(staged)["errors"] == ["This file is identical to the live version."]
    with pytest.raises(lib.LibraryError):
        lib.publish(staged)
    assert lib.active_release_id() == "initial"


@pytest.mark.parametrize("body, expected", [
    (V2_POLICY.replace("B|", "None|"), "Section B headings were not found"),
    ("None|prose|Policy Wording.\nA|prose|Short.", "Passage count changed a lot (5 → 2)"),
    (V2_POLICY.replace("A|table_row|", "A|prose|"), "No tables were detected"),
    (V2_POLICY.replace("Policy Wording", "Benefits Schedule"), 'don\'t mention "Policy Wording"'),
])
def test_suspicious_parses_raise_warnings(lib, body, expected):
    warnings = lib.check_report(stage(lib, body=body))["warnings"]
    assert any(expected in w for w in warnings), warnings


def test_future_effective_date_is_flagged(lib):
    future = (date.today() + timedelta(days=60)).isoformat()
    warnings = lib.check_report(stage(lib, effective=future))["warnings"]
    assert any("in the future" in w for w in warnings)


def test_unreadable_pdf_is_rejected_and_cleaned_up(lib):
    with pytest.raises(lib.LibraryError, match="Couldn't read this PDF"):
        stage(lib, body="BROKEN")
    staging = lib.library_dir() / "staging"
    assert not staging.exists() or not any(staging.iterdir())


def test_only_one_release_is_staged_at_a_time(lib):
    first = stage(lib)
    second = stage(lib, body=V2_POLICY.replace("90 days", "91 days"))
    assert not lib.release_dir(first, staged=True).exists()
    assert lib.release_dir(second, staged=True).exists()


def test_publish_makes_it_live_and_changes_the_cache_version(lib):
    version_before = cache.doc_version()
    staged = stage(lib)
    lib.publish(staged)
    assert lib.active_release_id() == staged
    assert not lib.release_dir(staged, staged=True).exists()
    assert "90 days" in lib.active_paths()[0].read_text()
    assert cache.doc_version() != version_before
    assert lib.history()[-1]["action"] == "publish" and lib.history()[-1]["version"] == "v2"


def test_roll_back_to_an_earlier_release(lib):
    version_before = cache.doc_version()
    staged = stage(lib)
    lib.publish(staged)
    lib.make_live("initial")
    assert lib.active_release_id() == "initial"
    assert cache.doc_version() == version_before
    assert [m["release_id"] for m in lib.list_releases()] == [staged, "initial"]


def test_discard_removes_the_staged_release(lib):
    staged = stage(lib)
    lib.discard_staged(staged)
    assert not lib.release_dir(staged, staged=True).exists()
    assert lib.active_release_id() == "initial"


def test_dashboard_review_and_publish(lib, tmp_path, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    import server.db as db

    monkeypatch.setattr(db, "DB_FILE", tmp_path / "conversations.db")
    db.init_db()
    staged = stage(lib)

    monkeypatch.setenv("ADMIN_PASSWORD", "s3cret")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py"), default_timeout=30)
    app.session_state.authed = True
    app.run()
    assert not app.exception
    assert "Check this PDF" in [b.label for b in app.button]

    app.session_state["staged_release"] = staged
    app.run()
    assert {m.label: m.value for m in app.metric}["Passages"] == "5"
    publish = next(b for b in app.button if b.label == "Publish")
    assert not publish.disabled
    publish.click().run()

    assert not app.exception
    assert lib.active_release_id() == staged
    assert any("Published Policy Wording v2" in s.value for s in app.success)
