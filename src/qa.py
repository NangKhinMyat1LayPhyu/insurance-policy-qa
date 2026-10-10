import argparse
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

import cache
import library
from index import chunks_to_documents, get_embedder
from ingest import DOC_LABELS
from terms import classify_question, search_terms

load_dotenv()

MODEL = "gemini-3.5-flash-lite"
VECTOR_K = 10
BM25_K = 10
TERMS_K = 5
MAX_TOKENS = 4096

INCOMPLETE_ANSWER = "Sorry, I couldn't put together a complete answer to that just now. Please try asking again."


SECTION_LABELS = {"A": "Section A / HomeCare", "B": "Section B / Health", None: "general (applies to both sections)"}

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


SYSTEM_PROMPT = """You answer questions about an insurance policy (the Kasturi HomeCare & Health Plan, a SYNTHETIC product used for testing only) using ONLY the context passages provided by the user.

The context passages come from four documents, and they do not carry equal authority:
- "Policy Wording" is the actual contract. It defines what is covered, the definitions of key terms, and the exclusions. This is the primary source for "is X covered" questions.
- "Benefits Schedule" states the amounts (limits, sub-limits, Excess, Co-insurance) for each tier (Silver/Gold/Platinum). A figure appearing here does NOT by itself confirm something is covered -- the Policy Wording's exclusions still apply on top of it. Where the two disagree on an amount, the Benefits Schedule wins; on whether something is covered at all, the Policy Wording governs.
- "Claims Procedure Guide" governs only the claims PROCESS -- which form to use, what documents are needed, time limits. It does not determine whether a loss is covered.
- "Product Disclosure Sheet" is an informal plain-language summary for prospective customers and is explicitly NOT part of the insurance contract. Never treat it as authoritative if it conflicts with the Policy Wording, the Benefits Schedule or the Claims Procedure Guide. Only use it for general "how does this product work" questions, never to confirm coverage or an amount.

Each passage is also labeled with which policy Section it belongs to, when known: "Section A" (HomeCare / property), "Section B" (Health / medical), or unlabeled (general terms that apply to both). SECTION A AND SECTION B ARE SEPARATE, NON-INTERCHANGEABLE PARTS OF THE POLICY WITH DIFFERENT MECHANICS -- Section A claims are subject to an Excess; Section B claims are subject to Co-insurance and count toward an Overall Annual Limit. A property/HomeCare question (damage to a home, its contents, theft, fire, water damage, flood) is a Section A matter -- never cite a Section B figure (Co-insurance, Overall Annual Limit) in that answer, even if a Section B passage was retrieved alongside it. A medical/health question is a Section B matter -- never cite Section A figures (Excess, Sum Insured, Buildings/Contents) in that answer. If you are not sure which section a passage belongs to, do not guess whether it applies -- state the uncertainty rather than mixing sections.

Rules:
- Use only the context passages given. If they do not contain the answer, say "The provided documents do not answer this." and stop. Never fall back on outside knowledge about insurance in general.
- Check exclusions BEFORE quoting a coverage amount. A benefit shown in the Benefits Schedule can still be excluded by the Policy Wording for a specific situation -- if an exclusion applies, say so clearly and do not lead with the dollar figure.
- Coverage and amounts often depend on the tier (Silver/Gold/Platinum) and on specific conditions (e.g. whether the Special Perils Extension is endorsed). If the question does not state the tier or a condition the answer depends on, say what's missing and ask for it rather than guessing or defaulting to a tier.
- Every sentence stating a fact must end with a citation: (Source: <document name>, page <N>). Use the document names exactly as given above.
- Never cite the Product Disclosure Sheet for whether something is covered, for an exclusion, or for an amount -- cite the Policy Wording or Benefits Schedule passage instead. If only the Product Disclosure Sheet mentions it, say the contract documents provided do not confirm it.
- This is not a claims decision -- if the situation sounds like an actual claim, note that briefly at the end.
- Be concise."""


HISTORY_TURNS = 3
HISTORY_ANSWER_CHARS = 600

REWRITE_PROMPT = """You rewrite a customer's latest message to an insurance support bot so it can be understood WITHOUT the earlier conversation.

Rules:
- Output ONE standalone question and nothing else -- no preamble, no quotes, no answer.
- Carry over every detail from the conversation that the latest message depends on: the plan tier (Silver/Gold/Platinum), the peril or benefit being discussed (e.g. Escape of Water, Flood, hospitalisation), add-ons such as the Special Perils Extension, and what happened.
- Resolve words like "that", "it", "this", "the same" into what they refer to.
- If the latest message is already a complete question on its own, or changes to an unrelated topic, return it unchanged.
- Never add facts the customer did not state, and never answer the question."""


def rewrite_with_history(question, history, model=MODEL):
    if not history:
        return question

    transcript = "\n\n".join(
        f"Customer: {q}\nBot: {a[:HISTORY_ANSWER_CHARS]}" for q, a in history
    )
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=f"Conversation so far:\n\n{transcript}\n\nLatest message: {question}",
        config=types.GenerateContentConfig(
            system_instruction=REWRITE_PROMPT,
            max_output_tokens=1024,
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    finish_reason = response.candidates[0].finish_reason if response.candidates else None
    rewritten = (response.text or "").strip().strip('"')
    if finish_reason != types.FinishReason.STOP or not rewritten:
        return question
    return rewritten


@lru_cache(maxsize=4)
def _load_store(index_dir):
    return FAISS.load_local(index_dir, get_embedder(), allow_dangerous_deserialization=True)


@lru_cache(maxsize=4)
def _load_bm25(chunks_file):
    raw_chunks = json.loads(Path(chunks_file).read_text(encoding="utf-8"))
    retriever = BM25Retriever.from_documents(chunks_to_documents(raw_chunks))
    retriever.k = BM25_K
    return retriever


def load_store(index_dir):
    if not (Path(index_dir) / "index.faiss").exists():
        raise SystemExit(f"No FAISS index in {index_dir}/ — run `python src/index.py` first.")
    return _load_store(str(index_dir))


def load_bm25(chunks_file):
    if not Path(chunks_file).exists():
        raise SystemExit(f"No {chunks_file} — run `python src/ingest.py` first.")
    return _load_bm25(str(chunks_file))


def retrieve(question, question_terms=None, terms_k=None, release=None):
    classification = question_terms if question_terms is not None else classify_question(question)
    terms = search_terms(classification)
    search_query = f"{question} {terms}".strip()
    terms_k = TERMS_K if terms_k is None else terms_k

    chunks_file, index_dir = library.paths_for(release) if release else library.active_paths()
    store = load_store(index_dir)
    bm25 = load_bm25(chunks_file)
    vector_hits = store.similarity_search_with_score(QUERY_PREFIX + search_query, k=VECTOR_K)
    keyword_hits = bm25.invoke(search_query)
    if terms and terms_k:
        vector_hits += store.similarity_search_with_score(QUERY_PREFIX + terms, k=terms_k)
        keyword_hits += bm25.invoke(terms)[:terms_k]

    seen = set()
    merged = []
    for doc, score in vector_hits:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append((doc, "vector", score))
    for doc in keyword_hits:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append((doc, "keyword", None))
    return merged


def format_context(hits):
    blocks = []
    for doc, _method, _score in hits:
        label = DOC_LABELS.get(doc.metadata["doc_type"], doc.metadata["doc_type"])
        section = SECTION_LABELS.get(doc.metadata.get("section"), "general (applies to both sections)")
        blocks.append(f'(document: "{label}", page: {doc.metadata["page"]}, section: {section})\n{doc.page_content}')
    return "\n\n---\n\n".join(blocks)


def answer(question, model=MODEL, use_cache=True, cache_threshold=None, release=None):
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise SystemExit("No GEMINI_API_KEY found. Add it to .env (see .env.example).")

    question_terms = classify_question(question)

    if release is not None:
        use_cache = False

    if use_cache:
        result = cache.lookup(question, threshold=cache_threshold or cache.DEFAULT_THRESHOLD, question_terms=question_terms)
        if result["hit"]:
            return result["answer"], [], {**result, **question_terms}

    hits = retrieve(question, question_terms=question_terms, release=release)

    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=f"Question: {question}\n\nContext passages:\n\n{format_context(hits)}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=MAX_TOKENS,
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    text = response.text
    finish_reason = response.candidates[0].finish_reason if response.candidates else None
    if finish_reason != types.FinishReason.STOP or not text:
        return INCOMPLETE_ANSWER, hits, {"hit": False, "incomplete": True, **question_terms}

    if use_cache:
        cache.store(question, text, question_terms=question_terms)

    return text, hits, {"hit": False, **question_terms}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Insurance document Q&A (Stage 1+2).")
    parser.add_argument("question", help="the question to answer")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--no-cache", action="store_true", help="skip the semantic cache")
    args = parser.parse_args(argv)

    try:
        text, hits, cache_info = answer(args.question, model=args.model, use_cache=not args.no_cache)
    except errors.ClientError as e:
        sys.exit(f"Gemini rejected the request (check API key / model in .env): {e}")
    except errors.APIError as e:
        sys.exit(f"Gemini API error: {e}")

    if cache_info.get("hit"):
        print(f"[cache hit, similarity {cache_info['similarity']:.3f}, matched: \"{cache_info['matched_question']}\"]\n")
    print(text)
    print(f"\nRetrieved chunks ({len(hits)}):")
    for doc, method, score in hits:
        label = DOC_LABELS.get(doc.metadata["doc_type"], doc.metadata["doc_type"])
        score_str = f"{score:.3f}" if score is not None else "  -  "
        print(f"  [{method:7s} {score_str}] {label} p.{doc.metadata['page']} ({doc.metadata['kind']})")


if __name__ == "__main__":
    main()
