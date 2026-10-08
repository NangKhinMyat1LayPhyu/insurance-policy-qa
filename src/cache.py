import hashlib
import json
import time
from pathlib import Path

import numpy as np

from index import CHUNKS_FILE, get_embedder
from terms import classify_question

ROOT = Path(__file__).resolve().parents[1]
CACHE_FILE = ROOT / "data" / "cache.json"
CACHE_LOG_FILE = ROOT / "data" / "cache_log.jsonl"

DEFAULT_THRESHOLD = 0.98


def doc_version():
    if not CHUNKS_FILE.exists():
        return "no-chunks"
    return hashlib.sha256(CHUNKS_FILE.read_bytes()).hexdigest()[:16]


def load_cache():
    if not CACHE_FILE.exists():
        return []
    return json.loads(CACHE_FILE.read_text(encoding="utf-8"))


def save_cache(entries):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def log_decision(question, matched_question, similarity, hit, threshold):
    CACHE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "matched_question": matched_question,
        "similarity": similarity,
        "hit": hit,
        "threshold": threshold,
    }
    with CACHE_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def category_match(a, b):
    if not a.get("category") or not b.get("category"):
        return False
    if a["category"] != b["category"]:
        return False
    if a.get("tier") and b.get("tier") and a["tier"] != b["tier"]:
        return False
    return True


def find_best_match(question_embedding, classification, entries, version):
    best, best_sim = None, 0.0
    for entry in entries:
        if entry["doc_version"] != version:
            continue
        entry_classification = {"category": entry.get("category"), "tier": entry.get("tier")}
        if not category_match(classification, entry_classification):
            continue
        sim = cosine_sim(question_embedding, entry["embedding"])
        if sim > best_sim:
            best_sim = sim
            best = entry
    return best, best_sim


def lookup(question, threshold=DEFAULT_THRESHOLD, log=True, question_terms=None):
    entries = load_cache()
    q_emb = get_embedder().embed_query(question)
    classification = question_terms if question_terms is not None else classify_question(question)
    best, best_sim = find_best_match(q_emb, classification, entries, doc_version())

    hit = best is not None and best_sim >= threshold
    result = {
        "hit": hit,
        "answer": best["answer"] if hit else None,
        "matched_question": best["question"] if best else None,
        "similarity": best_sim,
    }
    if log:
        log_decision(question, result["matched_question"], best_sim, hit, threshold)
    return result


def store(question, answer, question_terms=None):
    classification = question_terms if question_terms is not None else classify_question(question)
    entries = load_cache()
    entries.append(
        {
            "question": question,
            "embedding": get_embedder().embed_query(question),
            "category": classification.get("category"),
            "tier": classification.get("tier"),
            "answer": answer,
            "doc_version": doc_version(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    save_cache(entries)
