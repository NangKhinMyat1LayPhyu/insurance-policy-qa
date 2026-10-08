import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import qa

CONVERSATIONS_FILE = ROOT / "eval" / "memory_conversations.json"
RESULTS_FILE = ROOT / "eval" / "results" / "memory.json"
PAUSE_SECONDS = 10


def ask(question):
    text, _hits, _info = qa.answer(question, use_cache=False)
    time.sleep(PAUSE_SECONDS)
    return text


def passes(answer, expect):
    return all(fact.lower() in answer.lower() for fact in expect)


def main():
    conversations = json.loads(CONVERSATIONS_FILE.read_text(encoding="utf-8"))
    rows = []
    for i, conv in enumerate(conversations, start=1):
        first, follow_up = conv["turns"]
        expect = conv.get("expect", [])
        forbidden = conv.get("rewrite_must_not_mention", [])

        without_memory = ask(follow_up)
        first_answer = ask(first)
        rewritten = qa.rewrite_with_history(follow_up, [(first, first_answer)])
        with_memory = ask(rewritten)

        row = {
            "follow_up": follow_up,
            "rewritten": rewritten,
            "without_memory": {"answer": without_memory, "pass": passes(without_memory, expect)},
            "with_memory": {"answer": with_memory, "pass": passes(with_memory, expect)},
        }
        if forbidden:
            row["rewrite_kept_topic_separate"] = not any(w in rewritten.lower() for w in forbidden)
        rows.append(row)

        extra = f"  topic kept separate: {row['rewrite_kept_topic_separate']}" if forbidden else ""
        print(f"[{i}/{len(conversations)}] {follow_up}\n"
              f"    rewritten: {rewritten}\n"
              f"    without memory: {'PASS' if row['without_memory']['pass'] else 'FAIL'}"
              f"   with memory: {'PASS' if row['with_memory']['pass'] else 'FAIL'}{extra}", flush=True)

    scored = [r for r, c in zip(rows, conversations) if c.get("expect")]
    summary = {name: f"{sum(r[name]['pass'] for r in scored)}/{len(scored)}" for name in ("without_memory", "with_memory")}
    print(f"\nFollow-ups answered correctly -- without memory: {summary['without_memory']}, with memory: {summary['with_memory']}")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps({"summary": summary, "conversations": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {RESULTS_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
