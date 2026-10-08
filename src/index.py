from functools import lru_cache
from pathlib import Path
import json

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "data" / "chunks.json"
INDEX_DIR = ROOT / "data" / "index"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32


class BGEEmbeddings(Embeddings):
    def __init__(self, model_name=MODEL_NAME, batch_size=BATCH_SIZE):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def embed_documents(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embedder():
    return BGEEmbeddings()


def main():
    if not CHUNKS_FILE.exists():
        raise SystemExit(f"{CHUNKS_FILE} not found — run `python src/ingest.py` first.")
    raw_chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))

    documents = [
        Document(
            page_content=c["text"],
            metadata={
                "doc_type": c["doc_type"],
                "source_file": c["source_file"],
                "page": c["page"],
                "kind": c["kind"],
                "section": c.get("section"),
            },
        )
        for c in raw_chunks
    ]

    print(f"Embedding {len(documents)} chunks with {MODEL_NAME} (batch_size={BATCH_SIZE})...")
    store = FAISS.from_documents(documents, get_embedder())

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))
    print(f"Saved FAISS index to {INDEX_DIR.relative_to(ROOT)}/ ({len(documents)} vectors)")


if __name__ == "__main__":
    main()
