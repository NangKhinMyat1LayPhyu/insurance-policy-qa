import fcntl
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pdfplumber

import ingest
from index import CHUNKS_FILE, INDEX_DIR, build_index

MAX_PASSAGE_CHANGE = 0.30


class LibraryError(Exception):
    pass


def library_dir():
    value = os.getenv("LIBRARY_DIR")
    return Path(value) if value else None


@contextmanager
def _lock():
    root = library_dir()
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield root
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _new_id():
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _page_count(path):
    with pdfplumber.open(path) as pdf:
        return len(pdf.pages)


def _write_json(path, data):
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _set_active(root, release_id):
    tmp = root / "active.tmp"
    tmp.write_text(release_id, encoding="utf-8")
    os.replace(tmp, root / "active")


def _log(root, **event):
    with (root / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **event}) + "\n")


def _doc_entry(pdf_path, version, effective_date):
    return {
        "file": Path(pdf_path).name,
        "version": version,
        "effective_date": effective_date,
        "pages": _page_count(pdf_path),
        "sha256": _sha256(pdf_path),
    }


def active_release_id():
    root = library_dir()
    if not root or not (root / "active").exists():
        return None
    return (root / "active").read_text(encoding="utf-8").strip()


def release_dir(release_id, staged=False):
    return library_dir() / ("staging" if staged else "releases") / release_id


def active_paths():
    release_id = active_release_id()
    if release_id is None:
        return CHUNKS_FILE, INDEX_DIR
    return paths_for(release_dir(release_id))


def paths_for(directory):
    directory = Path(directory)
    return directory / "chunks.json", directory / "index"


def active_chunks_file():
    return active_paths()[0]


def read_manifest(directory):
    return json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))


def load_chunks(directory):
    return json.loads(paths_for(directory)[0].read_text(encoding="utf-8"))


def ensure_initialized():
    if library_dir() is None:
        return
    with _lock() as root:
        if (root / "active").exists():
            return
        release_id = "initial"
        target = root / "releases" / release_id
        if target.exists():
            shutil.rmtree(target)
        (target / "documents").mkdir(parents=True)
        documents = {}
        for doc in ingest.DOCUMENTS:
            dest = target / "documents" / doc["file"]
            shutil.copy2(ingest.ROOT / doc["file"], dest)
            documents[doc["doc_type"]] = _doc_entry(dest, "initial", None)
        if CHUNKS_FILE.exists() and (INDEX_DIR / "index.faiss").exists():
            shutil.copy2(CHUNKS_FILE, target / "chunks.json")
            shutil.copytree(INDEX_DIR, target / "index")
        else:
            chunks = ingest.build_chunks({k: target / "documents" / v["file"] for k, v in documents.items()})
            _write_json(target / "chunks.json", chunks)
            build_index(chunks, target / "index")
        _write_json(target / "manifest.json", {
            "release_id": release_id, "based_on": None, "changed": None, "note": "Initial documents",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "published": time.strftime("%Y-%m-%d %H:%M:%S"),
            "published_ts": time.time(), "documents": documents,
        })
        _set_active(root, release_id)
        _log(root, action="initialise", release_id=release_id)


def _safe_filename(name):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "document"
    return f"{stem}.pdf"


def stage_replacement(doc_type, pdf_bytes, filename, version, effective_date, note=""):
    if library_dir() is None:
        raise LibraryError("Document management needs LIBRARY_DIR to be set.")
    if doc_type not in ingest.DOC_LABELS:
        raise LibraryError(f"Unknown document type: {doc_type}")
    base_id = active_release_id()
    if base_id is None:
        raise LibraryError("No live release yet.")

    with _lock() as root:
        shutil.rmtree(root / "staging", ignore_errors=True)
        release_id = _new_id()
        target = release_dir(release_id, staged=True)
        (target / "documents").mkdir(parents=True)

        base = release_dir(base_id)
        manifest = read_manifest(base)
        documents = {}
        for dtype, entry in manifest["documents"].items():
            if dtype != doc_type:
                shutil.copy2(base / "documents" / entry["file"], target / "documents" / entry["file"])
                documents[dtype] = entry
        new_pdf = target / "documents" / _safe_filename(filename)
        if new_pdf.exists():
            new_pdf = new_pdf.with_name(f"{new_pdf.stem}_{release_id}.pdf")
        new_pdf.write_bytes(pdf_bytes)

        try:
            documents[doc_type] = _doc_entry(new_pdf, version, effective_date)
            chunks = ingest.build_chunks({k: target / "documents" / v["file"] for k, v in documents.items()})
        except Exception as e:
            shutil.rmtree(target, ignore_errors=True)
            raise LibraryError(f"Couldn't read this PDF ({type(e).__name__}: {e}).") from e

        _write_json(target / "chunks.json", chunks)
        build_index(chunks, target / "index")
        _write_json(target / "manifest.json", {
            "release_id": release_id, "based_on": base_id, "changed": doc_type, "note": note,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "published": None, "documents": documents,
        })
    return release_id


def discard_staged(release_id):
    shutil.rmtree(release_dir(release_id, staged=True), ignore_errors=True)


def _summary(chunks, doc_type):
    doc_chunks = [c for c in chunks if c["doc_type"] == doc_type]
    return {
        "passages": len(doc_chunks),
        "table_rows": sum(1 for c in doc_chunks if c["kind"] == "table_row"),
        "sections": sorted({c["section"] for c in doc_chunks if c["section"]}),
        "pages_with_text": len({c["page"] for c in doc_chunks}),
    }


def _sentences(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = " ".join(ingest.clean_prose(page.extract_text() or "") for page in pdf.pages)
    text = re.sub(r"\s+", " ", text)
    seen, out = set(), []
    for sentence in re.split(r"(?<=[.;])\s+(?=[A-Z0-9(\"“])", text):
        sentence = sentence.strip()
        if len(sentence) > 3 and sentence not in seen:
            seen.add(sentence)
            out.append(sentence)
    return out


def _title_found(chunks, doc_type):
    first_page = " ".join(c["text"] for c in chunks if c["doc_type"] == doc_type and c["page"] <= 2)
    return ingest.DOC_LABELS[doc_type].lower() in first_page.lower()


def check_report(release_id):
    staged = release_dir(release_id, staged=True)
    manifest = read_manifest(staged)
    base = release_dir(manifest["based_on"])
    doc_type = manifest["changed"]
    old_entry, new_entry = read_manifest(base)["documents"][doc_type], manifest["documents"][doc_type]
    old_chunks, new_chunks = load_chunks(base), load_chunks(staged)
    old, new = _summary(old_chunks, doc_type), _summary(new_chunks, doc_type)
    label = ingest.DOC_LABELS[doc_type]

    errors, warnings = [], []
    if new_entry["sha256"] == old_entry["sha256"]:
        errors.append("This file is identical to the live version.")
    if new["passages"] == 0:
        errors.append("No text could be read from this PDF. It may be a scanned image.")
    if new["passages"] and not _title_found(new_chunks, doc_type):
        warnings.append(f'The first pages don\'t mention "{label}". Check this is the right document.')
    if old["passages"] and abs(new["passages"] - old["passages"]) / old["passages"] > MAX_PASSAGE_CHANGE:
        warnings.append(f"Passage count changed a lot ({old['passages']} → {new['passages']}). Check the PDF was read correctly.")
    lost = sorted(set(old["sections"]) - set(new["sections"]))
    if lost:
        warnings.append(f"Section {', '.join(lost)} headings were not found, so those passages can't be told apart from the other section.")
    if old["table_rows"] and not new["table_rows"]:
        warnings.append("No tables were detected, but the live version has them. Figures may have been read as plain text.")
    effective = new_entry.get("effective_date")
    if effective and effective > date.today().isoformat():
        warnings.append(f"Effective date {effective} is in the future. Once published, customers on current "
                        "policies will get answers from this version.")

    old_sentences = _sentences(base / "documents" / old_entry["file"])
    new_sentences = _sentences(staged / "documents" / new_entry["file"])
    old_set, new_set = set(old_sentences), set(new_sentences)

    return {
        "doc_type": doc_type,
        "label": label,
        "old": {**old, "pages": old_entry["pages"], "version": old_entry["version"], "file": old_entry["file"]},
        "new": {**new, "pages": new_entry["pages"], "version": new_entry["version"], "file": new_entry["file"],
                "effective_date": effective},
        "errors": errors,
        "warnings": warnings,
        "removed": [s for s in old_sentences if s not in new_set],
        "added": [s for s in new_sentences if s not in old_set],
    }


def publish(release_id):
    staged = release_dir(release_id, staged=True)
    report = check_report(release_id)
    if report["errors"]:
        raise LibraryError("Can't publish: " + " ".join(report["errors"]))
    with _lock() as root:
        manifest = read_manifest(staged)
        manifest["published"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest["published_ts"] = time.time()
        _write_json(staged / "manifest.json", manifest)
        target = release_dir(release_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), str(target))
        _set_active(root, release_id)
        entry = manifest["documents"][manifest["changed"]]
        _log(root, action="publish", release_id=release_id, document=manifest["changed"],
             version=entry["version"], effective_date=entry["effective_date"], note=manifest["note"])


def make_live(release_id):
    if not (release_dir(release_id) / "manifest.json").exists():
        raise LibraryError(f"No published release {release_id}.")
    with _lock() as root:
        _set_active(root, release_id)
        _log(root, action="make live", release_id=release_id)


def list_releases():
    root = library_dir()
    if root is None or not (root / "releases").exists():
        return []
    manifests = [read_manifest(d) for d in (root / "releases").iterdir() if (d / "manifest.json").exists()]
    return sorted(manifests, key=lambda m: m.get("published_ts") or 0, reverse=True)


def history():
    root = library_dir()
    path = root / "history.jsonl" if root else None
    if not path or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
