"""PDF ingestion and persistent Chroma retrieval."""
from __future__ import annotations
import hashlib
import re
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


class ChromaDefaultEmbeddings(Embeddings):
    """Adapter around Chroma's bundled local ONNX MiniLM embedding function."""
    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, self._fn([text])[0]))


def paper_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:24]


def load_pdf(data: bytes) -> list[Document]:
    """Extract text from PDF while retaining one-based page metadata."""
    import io
    reader = PdfReader(io.BytesIO(data))
    docs: list[Document] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"page": number}))
    if not docs:
        raise ValueError("No extractable text was found. This PDF may be scanned; use a text-based PDF.")
    return docs


def load_text(text: str) -> list[Document]:
    """Load plain text as a single logical page for the built-in demo."""
    if not text.strip():
        raise ValueError("The sample paper is empty.")
    # Approximate page labels for a readable demo trace.
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    docs, page, count = [], 1, 0
    for para in paras:
        if count > 1800:
            page += 1
            count = 0
        docs.append(Document(page_content=para, metadata={"page": page}))
        count += len(para)
    return docs


def split_documents(docs: Iterable[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    chunks = splitter.split_documents(list(docs))
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def get_vector_store(data: bytes, docs: list[Document], chunk_size: int = 1000, persist_dir: str = "./chroma_db"):
    """Create or reuse a per-paper persistent Chroma collection."""
    from langchain_chroma import Chroma
    key = paper_hash(data)
    chunks = split_documents(docs, chunk_size=chunk_size)
    store = Chroma(collection_name=f"paper_{key}", embedding_function=ChromaDefaultEmbeddings(), persist_directory=persist_dir)
    # Avoid duplicate inserts when Streamlit reruns: inspect collection count.
    if store._collection.count() == 0:
        ids = [f"{key}_{i}" for i in range(len(chunks))]
        store.add_documents(chunks, ids=ids)
    return store, key, chunks


def retrieve(store, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
    """Return similarity-ranked chunks and Chroma distance scores."""
    return store.similarity_search_with_score(query, k=top_k)


def format_hits(hits: list[tuple[Document, float]]) -> str:
    return "\n\n".join(f"[p.{d.metadata.get('page', '?')} | chunk {d.metadata.get('chunk_id', '?')} | distance {score:.4f}]\n{d.page_content}" for d, score in hits)
