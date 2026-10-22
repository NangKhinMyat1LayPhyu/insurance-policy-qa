"""Parse the policy PDFs into clean chunks tagged with document, page and policy section.
Benefits-table rows keep their Silver / Gold / Platinum headers.
"""

import json
import re
from pathlib import Path

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "data" / "chunks.json"

TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 5,
    "join_tolerance": 5,
}

DOCUMENTS = [
    {"file": "01_Product_Disclosure_Sheet.pdf", "doc_type": "product_disclosure_sheet", "label": "Product Disclosure Sheet"},
    {"file": "02_Policy_Wording.pdf", "doc_type": "policy_wording", "label": "Policy Wording"},
    {"file": "03_Benefits_Schedule.pdf", "doc_type": "benefits_schedule", "label": "Benefits Schedule"},
    {"file": "04_Claims_Procedure_Guide.pdf", "doc_type": "claims_procedure_guide", "label": "Claims Procedure Guide"},
]
DOC_LABELS = {doc["doc_type"]: doc["label"] for doc in DOCUMENTS}

# Section A (property) and B (health) amounts must never be mixed, so chunks carry their section.
SECTION_MARKERS = [
    (re.compile(r"Part 3\s*.\s*Section A: HomeCare"), "A"),
    (re.compile(r"Part 4\s*.\s*Section B: Health"), "B"),
    (re.compile(r"5\.2 Exclusions applicable to Section A"), "A"),
    (re.compile(r"5\.3 Exclusions applicable to Section B"), "B"),
    (re.compile(r"Part A\s*.\s*Section A: HomeCare"), "A"),
    (re.compile(r"Part B\s*.\s*Section B: Health"), "B"),
    (re.compile(r"3\. Claims under Section A"), "A"),
    (re.compile(r"4\. Claims under Section B"), "B"),
    (re.compile(r"Part 6\s*.\s*General Conditions"), None),
    (re.compile(r"5\. Payment of claims"), None),
]


def detect_section(page_text, current):
    for pattern, section in SECTION_MARKERS:
        if pattern.search(page_text):
            current = section
    return current

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Repeated page headers and footers are removed so they don't dominate every chunk's embedding.
BOILERPLATE_PATTERNS = [
    re.compile(r"^L\s*U\s*M\s*I\s*N\s*A.*$", re.MULTILINE),
    re.compile(r"^Lumina Kasturi Assurance Berhad.*SYNTHETIC.*$", re.MULTILINE),
    re.compile(r"^LKA/HHP/.*Page \d+ of \d+$", re.MULTILINE),
    re.compile(r"^SYNTHETIC DOCUMENT.*$", re.MULTILINE),
]


def clean_cell(cell):
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", cell).strip()


def clean_prose(text):
    for pattern in BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def table_rows_as_chunks(table, doc_type, file, page_num, section):
    rows = table.extract()
    if not rows or len(rows) < 2:
        return []
    header = [clean_cell(c) for c in rows[0]]
    chunks = []
    for row in rows[1:]:
        cells = [clean_cell(c) for c in row]
        if not any(cells) or not cells[0]:
            continue
        label = cells[0]
        parts = [f"{header[i]}: {cells[i]}" for i in range(1, len(cells)) if i < len(header) and cells[i]]
        if not parts:
            continue
        text = f"{label} -- " + "; ".join(parts) + "."
        chunks.append(
            {"text": text, "doc_type": doc_type, "source_file": file, "page": page_num, "kind": "table_row", "section": section}
        )
    return chunks


def extract_document(path, doc_type, splitter):
    path = Path(path)
    file = path.name
    chunks = []
    current_section = None
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw_page_text = page.extract_text() or ""
            current_section = detect_section(raw_page_text, current_section)

            tables = page.find_tables(table_settings=TABLE_SETTINGS)
            bboxes = [t.bbox for t in tables]

            for table in tables:
                chunks.extend(table_rows_as_chunks(table, doc_type, file, page_num, current_section))

            def not_in_table(obj, bboxes=bboxes):
                for (x0, top, x1, bottom) in bboxes:
                    if obj["top"] >= top - 1 and obj["bottom"] <= bottom + 1:
                        return False
                return True

            prose_page = page.filter(not_in_table) if bboxes else page
            prose_text = clean_prose(prose_page.extract_text() or "")
            for chunk_text in splitter.split_text(prose_text):
                if chunk_text.strip():
                    chunks.append(
                        {
                            "text": chunk_text,
                            "doc_type": doc_type,
                            "source_file": file,
                            "page": page_num,
                            "kind": "prose",
                            "section": current_section,
                        }
                    )
    return chunks


def build_chunks(doc_paths):
    """`doc_paths` maps each doc_type to its PDF. Returns all chunks, in
    DOCUMENTS order. Used both for the repo's own PDFs (main) and for a set
    of documents staged from the admin dashboard (library.py).
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_chunks = []
    for doc in DOCUMENTS:
        all_chunks.extend(extract_document(doc_paths[doc["doc_type"]], doc["doc_type"], splitter))
    return all_chunks


def default_doc_paths():
    return {doc["doc_type"]: ROOT / doc["file"] for doc in DOCUMENTS}


def main():
    all_chunks = build_chunks(default_doc_paths())
    for doc in DOCUMENTS:
        doc_chunks = [c for c in all_chunks if c["doc_type"] == doc["doc_type"]]
        n_tables = sum(1 for c in doc_chunks if c["kind"] == "table_row")
        print(f"  {doc['file']:35s} {len(doc_chunks):4d} chunks ({n_tables} table rows, {len(doc_chunks) - n_tables} prose)")

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_FILE.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"\n{len(all_chunks)} total chunks -> {CHUNKS_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
