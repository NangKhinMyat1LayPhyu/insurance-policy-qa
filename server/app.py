import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from google.genai import errors as genai_errors
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import qa

from server.db import init_db, log_message, recent_turns
from server.escalation import flag_reason


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Kasturi HomeCare & Health Plan -- Support Widget API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


class ChatRequest(BaseModel):
    session_id: str
    message: str


class Source(BaseModel):
    document: str
    page: int
    section: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    cache_hit: bool
    similarity: Optional[float] = None
    sources: list[Source]
    needs_human: bool = False


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/static/demo-insurer-site.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        history = recent_turns(req.session_id, qa.HISTORY_TURNS)
        standalone = qa.rewrite_with_history(req.message, history)
        text, hits, cache_info = qa.answer(standalone)
    except genai_errors.APIError as e:
        status = 503 if e.code == 429 else 502
        raise HTTPException(status_code=status, detail="The answer service is busy. Please try again in a minute.")

    sources = []
    seen = set()
    for doc, _method, _score in hits:
        label = qa.DOC_LABELS.get(doc.metadata["doc_type"], doc.metadata["doc_type"])
        page = doc.metadata["page"]
        key = (label, page)
        if key not in seen:
            seen.add(key)
            sources.append(Source(document=label, page=page, section=doc.metadata.get("section")))

    reason = flag_reason(req.message, text, cache_info)

    log_message(
        session_id=req.session_id,
        question=req.message,
        answer=text,
        cache_hit=cache_info.get("hit", False),
        similarity=cache_info.get("similarity"),
        matched_question=cache_info.get("matched_question"),
        category=cache_info.get("category"),
        tier=cache_info.get("tier"),
        flag_reason=reason,
        standalone_question=standalone if standalone != req.message else None,
    )

    return ChatResponse(
        answer=text,
        cache_hit=cache_info.get("hit", False),
        similarity=cache_info.get("similarity"),
        sources=sources,
        needs_human=reason is not None,
    )
