"""Retrieval recall: for each question, does the passage that actually holds
the answer reach the model's context? Compares retrieval with and without
the canonical-terms-only pass (qa.retrieve()'s terms_k).
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import qa
from terms import classify_question

QUESTIONS_FILE = ROOT / "eval" / "retrieval_questions.json"
RESULTS_FILE = ROOT / "eval" / "results" / "retrieval_recall.json"
STRATEGIES = {"question + terms (before)": 0, "+ terms-only pass (after)": qa.TERMS_K}
PAUSE_SECONDS = 8


def rank_of(target, hits):
    return next((i + 1 for i, (doc, _m, _s) in enumerate(hits) if target in doc.page_content), None)


def main():
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    rows = []
    for i, item in enumerate(questions):
        classification = classify_question(item["question"])
        row = {"question": item["question"], "classified_as": classification}
        for name, terms_k in STRATEGIES.items():
            hits = qa.retrieve(item["question"], question_terms=classification, terms_k=terms_k)
            row[name] = {"found_at": rank_of(item["target"], hits), "context_size": len(hits)}
        rows.append(row)
        print(f"[{i + 1}/{len(questions)}] {classification.get('category')!s:35s} "
              + "  ".join(f"{name}: {'found' if row[name]['found_at'] else 'MISSED'}" for name in STRATEGIES))
        time.sleep(PAUSE_SECONDS)

    print()
    summary = {}
    for name in STRATEGIES:
        found = sum(1 for r in rows if r[name]["found_at"])
        avg_context = sum(r[name]["context_size"] for r in rows) / len(rows)
        summary[name] = {"recall": f"{found}/{len(rows)}", "avg_context_passages": round(avg_context, 1)}
        print(f"{name:28s} recall {found}/{len(rows)}   avg passages in context {avg_context:.1f}")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps({"summary": summary, "questions": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {RESULTS_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
