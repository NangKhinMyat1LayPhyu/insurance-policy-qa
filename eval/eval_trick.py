"""Trick questions: does the bot get the three hardest questions right, every time?"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from google.genai import errors as genai_errors

import qa

QUESTIONS_FILE = ROOT / "eval" / "trick_questions.json"
RESULTS_FILE = ROOT / "eval" / "results" / "trick_questions.json"
PAUSE_SECONDS = 10
RETRY_SECONDS = 60


def ask(question):
    for attempt in range(3):
        try:
            text, _hits, info = qa.answer(question, use_cache=False)
            time.sleep(PAUSE_SECONDS)
            return text, bool(info.get("incomplete"))
        except genai_errors.APIError as e:
            if attempt == 2:
                raise
            print(f"  Gemini busy ({e.code}); waiting {RETRY_SECONDS} s")
            time.sleep(RETRY_SECONDS)


def check(answer, incomplete, spec):
    text = answer.lower()
    missing = [p for p in spec["all"] if p.lower() not in text]
    missing += [" / ".join(group) for group in spec["any"] if not any(p.lower() in text for p in group)]
    return not incomplete and not missing, missing


def main():
    config = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    runs = config["runs_per_question"]
    saved = None
    if "--rescore" in sys.argv:
        saved = {q["question"]: q["runs"] for q in json.loads(RESULTS_FILE.read_text(encoding="utf-8"))["questions"]}
    rows, passed, total = [], 0, 0
    for spec in config["questions"]:
        results = []
        for i in range(1, runs + 1):
            if saved is not None:
                old = saved[spec["question"]][i - 1]
                answer, incomplete = old["answer"], old["incomplete"]
            else:
                answer, incomplete = ask(spec["question"])
            ok, missing = check(answer, incomplete, spec)
            results.append({"run": i, "pass": ok, "incomplete": incomplete, "missing": missing, "answer": answer})
            print(f"{spec['name'][:45]:45s} run {i}: {'pass' if ok else 'FAIL ' + str(missing)}")
        n_pass = sum(r["pass"] for r in results)
        passed, total = passed + n_pass, total + runs
        rows.append({"name": spec["name"], "question": spec["question"], "passed": f"{n_pass}/{runs}", "runs": results})

    summary = {"passed": f"{passed}/{total}", "per_question": {r["name"]: r["passed"] for r in rows},
               "model": qa.MODEL, "cache": "off"}
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps({"summary": summary, "questions": rows}, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    print(f"\nTrick questions answered correctly: {passed} / {total}")
    print(f"Saved: {RESULTS_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
