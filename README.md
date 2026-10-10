# Insurance Policy Q&A Bot

An AI support bot that answers customer questions about an insurance policy using **only the insurer's own documents**, citing the document and page for every fact. It comes with an admin dashboard for staff.

<img src="image/chat.png" alt="Chat widget answering that a phone stolen from a car is not covered, citing the exclusion clauses" width="700">

> The four policy PDFs are synthetic: an invented insurer and product modelled on real Malaysian general-insurance documents.

## How it answers

Insurance answers depend on several documents with different authority. The closest-matching passage is often the wrong one:

| Customer asks | Naive answer | Correct answer |
|---|---|---|
| "My house flooded. Gold tier, no Special Perils Extension. Covered?" | Yes, 50% of Sum Insured | **Not covered.** Flood needs the Extension (Clause A4.1). |
| "My phone was stolen from my car in the driveway." | Quotes the phone limit | **Not covered.** Exclusion 5.2.6 applies. |
| "My kitchen pipe burst and damaged my floor. Silver tier." | Finds nothing | **Covered:** RM25,000 limit, RM1,000 Excess |

## How it works

```mermaid
flowchart LR
    Q[Question] --> M[Rewrite follow-ups<br/>using recent turns]
    M --> C[Classify into<br/>policy term + tier]
    C --> K{Semantic<br/>cache}
    K -- hit --> A[Answer + citations]
    K -- miss --> R[Hybrid search<br/>FAISS + BM25]
    R --> G[Gemini, with<br/>document-precedence rules]
    G --> A
    A --> F{Needs a human?}
    F --> D[(Log)] --> B[Admin dashboard]
```

- **Table-aware PDF parsing:** every benefits-table row keeps its Silver/Gold/Platinum headers, and every passage is tagged with its document and policy section.
- **Grounded answers:** exclusions are checked before amounts, the contract outranks the summary, and every fact is cited.

## Results

| | |
|---|---|
| Trick questions answered correctly | **21 / 21** runs |
| Answer passage found by search | **13 / 13** questions |
| Wrong cached answers on near-miss questions | **0%** |
| Follow-up questions answered correctly | **4 / 4** (1 / 4 without memory) |

## Highlights

- **Fixed cut-off answers** by tracing them to thinking tokens using up the output budget. Unfinished answers are now never shown or cached.
- **Raised search recall from 12/13 to 13/13** by adding a second search on the policy term alone.
- **Made the cache safe** by matching on classified policy terms, since raw similarity failed on near-miss questions at every threshold.

## Admin dashboard

Headline numbers, a "needs a human" queue, searchable conversation transcripts, and topic insights.

<img src="image/dashboard-needs-human.png" alt="Dashboard headline numbers and the needs-a-human queue" width="700">

**Updating documents:** an admin uploads a new version of a policy PDF, reviews a check report and a side-by-side comparison of answers, then publishes. The bot switches without a restart, and earlier versions can be restored in one click.

<img src="image/documents-compare.png" alt="The same questions answered by the live and the new version of the Policy Wording" width="700">

## Run it

```bash
cp .env.example .env        # add your GEMINI_API_KEY
docker compose up --build
```

Chat widget: http://localhost:8000 · Dashboard: http://localhost:8501 · Sample data: `python3 scripts/seed_demo.py`

## Tech stack

Python · Gemini · sentence-transformers · FAISS · BM25 · LangChain · pdfplumber · FastAPI · SQLite · Streamlit · Docker
