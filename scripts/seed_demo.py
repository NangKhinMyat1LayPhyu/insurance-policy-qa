import argparse
import json
import time
import urllib.error
import urllib.request
import uuid

CONVERSATIONS = [
    [
        "My kitchen pipe burst and water damaged my flooring. I'm Silver tier - am I covered, and how much?",
        "Which claim form do I use for that?",
    ],
    ["My house flooded in the monsoon. I'm Gold tier, no Special Perils Extension. Am I covered?"],
    ["My phone was stolen from my car in the driveway - what do I claim and how much?"],
    [
        "How long is the waiting period before I can claim for hospitalisation?",
        "Do you cover my car if I crash it?",
    ],
    ["What is the Excess for Escape of Water on the Silver tier?"],
    ["What is the Excess for Escape of Water on the Silver tier?"],
    ["I've been waiting weeks on my claim. Can I talk to a real person please?"],
]

PAUSE_SECONDS = 10


def ask(api_url, session_id, question):
    body = json.dumps({"session_id": session_id, "message": question}).encode()
    request = urllib.request.Request(f"{api_url}/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()

    total = sum(len(c) for c in CONVERSATIONS)
    n = 0
    for conversation in CONVERSATIONS:
        session_id = str(uuid.uuid4())
        for question in conversation:
            n += 1
            print(f"[{n}/{total}] {question}")
            try:
                reply = ask(args.api_url, session_id, question)
            except urllib.error.HTTPError as e:
                print(f"    -> HTTP {e.code}: {e.read().decode()[:200]}")
            except urllib.error.URLError as e:
                raise SystemExit(f"Can't reach {args.api_url} ({e.reason}). Is the app running? Try `docker compose up`.")
            else:
                notes = ["cache hit" if reply["cache_hit"] else "answered"]
                if reply.get("needs_human"):
                    notes.append("flagged for a human")
                print(f"    -> {', '.join(notes)}: {reply['answer'][:90]!r}...")
            if n < total:
                time.sleep(PAUSE_SECONDS)
    print("\nDone. Open the dashboard at http://localhost:8501")


if __name__ == "__main__":
    main()
