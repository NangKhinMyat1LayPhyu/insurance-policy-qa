import json
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cache import category_match, cosine_sim
from index import get_embedder
from terms import classify_question

DATASET_FILE = Path(__file__).parent / "cache_questions.json"
RESULTS_FILE = Path(__file__).parent / "results" / "cache_threshold_sweep.json"
TERM_CACHE_FILE = Path(__file__).parent / "results" / "term_cache.json"

THRESHOLDS = [round(0.80 + 0.01 * i, 2) for i in range(20)]


def load_dataset():
    return json.loads(DATASET_FILE.read_text(encoding="utf-8"))


def all_questions(dataset):
    texts = set()
    for group in dataset["equivalent_groups"]:
        texts.update(group["questions"])
    for pair in dataset["adversarial_pairs"]:
        texts.add(pair["question_a"])
        texts.add(pair["question_b"])
    return sorted(texts)


def load_term_cache():
    if TERM_CACHE_FILE.exists():
        return json.loads(TERM_CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def get_classifications_for_all(questions):
    cache = load_term_cache()
    pending = [q for q in questions if not cache.get(q, {}).get("category")]
    for i, q in enumerate(pending):
        if i:
            time.sleep(4.5)
        classification = classify_question(q)
        cache[q] = classification
        print(f"  classified: {q[:50]:50s} -> {classification['category']!r} / {classification['tier']!r}")
        TERM_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        TERM_CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return {q: cache[q] for q in questions}


def embed_all(questions):
    print(f"Embedding {len(questions)} unique questions...")
    vectors = get_embedder().embed_documents(questions)
    return dict(zip(questions, vectors))


def build_pairs(dataset, vecs, classifications):
    def sim_and_gate(a, b):
        gate = category_match(classifications[a], classifications[b])
        return cosine_sim(vecs[a], vecs[b]), gate

    positive = []
    for group in dataset["equivalent_groups"]:
        for a, b in combinations(group["questions"], 2):
            positive.append(sim_and_gate(a, b))

    hard_negative = []
    for pair in dataset["adversarial_pairs"]:
        hard_negative.append(sim_and_gate(pair["question_a"], pair["question_b"]))

    general_negative = []
    groups = dataset["equivalent_groups"]
    for g1, g2 in combinations(groups, 2):
        for a in g1["questions"]:
            for b in g2["questions"]:
                general_negative.append(sim_and_gate(a, b))

    return positive, hard_negative, general_negative


def rate_at(pairs, threshold):
    if not pairs:
        return 0.0
    matches = sum(1 for sim, gate in pairs if gate and sim >= threshold)
    return matches / len(pairs)


def main():
    dataset = load_dataset()
    questions = all_questions(dataset)
    classifications = get_classifications_for_all(questions)
    vecs = embed_all(questions)
    positive, hard_negative, general_negative = build_pairs(dataset, vecs, classifications)

    n_gate_fail_positive = sum(1 for _, gate in positive if not gate)
    n_gate_fail_hard = sum(1 for _, gate in hard_negative if not gate)

    print(f"\n{len(positive)} equivalent (should-match) pairs "
          f"({n_gate_fail_positive} failed the category gate -- a real cost, see below)")
    print(f"{len(hard_negative)} adversarial (hard, must-NOT-match) pairs "
          f"({n_gate_fail_hard} correctly blocked by the category gate alone)")
    print(f"{len(general_negative)} general cross-topic (must-NOT-match) pairs\n")

    print(f"{'threshold':>9s}  {'hit rate':>9s}  {'false hit (hard)':>17s}  {'false hit (general)':>20s}")
    rows = []
    for t in THRESHOLDS:
        hit = rate_at(positive, t)
        false_hard = rate_at(hard_negative, t)
        false_general = rate_at(general_negative, t)
        rows.append({"threshold": t, "hit_rate": hit, "false_hit_hard": false_hard, "false_hit_general": false_general})
        print(f"{t:9.2f}  {hit:8.1%}  {false_hard:16.1%}  {false_general:19.1%}")

    zero_false_hard = [r for r in rows if r["false_hit_hard"] == 0.0]
    recommended = min(zero_false_hard, key=lambda r: r["threshold"]) if zero_false_hard else max(rows, key=lambda r: r["threshold"])

    print("\n" + "=" * 70)
    if zero_false_hard:
        print(f"Recommended threshold: {recommended['threshold']:.2f}")
        print(f"  -> {recommended['hit_rate']:.1%} hit rate, 0% false hits on the hard adversarial set")
        print("  Reasoning: lowest threshold with zero false hits on the pairs specifically")
        print("  designed to be hard to distinguish. A missed cache hit just means one extra")
        print("  full pipeline run (a fraction of a cent); a false hit risks serving a wrong")
        print("  coverage answer -- the asymmetry justifies giving up hit rate for safety.")
    else:
        print("No threshold in the sweep range achieved zero false hits on the hard set.")
        print(f"Highest threshold tested ({recommended['threshold']:.2f}) still had "
              f"{recommended['false_hit_hard']:.1%} false hits.")
    if n_gate_fail_positive:
        print(f"\nNote: {n_gate_fail_positive}/{len(positive)} genuinely equivalent pairs will NEVER "
              "be cache hits at ANY threshold now, because the category gate itself blocked them "
              "(e.g. classify_question() landed two paraphrases on different categories, or on "
              "different explicit tiers). That's the real cost of the gate, not a free fix.")
    print("=" * 70)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(
            {
                "sweep": rows,
                "recommended_threshold": recommended["threshold"],
                "n_positive": len(positive),
                "n_positive_gate_failed": n_gate_fail_positive,
                "n_hard_negative": len(hard_negative),
                "n_hard_negative_gate_blocked": n_gate_fail_hard,
                "n_general_negative": len(general_negative),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nfull results -> {RESULTS_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
