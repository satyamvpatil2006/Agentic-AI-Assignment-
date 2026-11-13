"""Offline smoke tests for ingestion, Chroma retrieval, and intent parsing."""
from __future__ import annotations
import tempfile
from pathlib import Path
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
from agents import IntentRouter
from rag import load_text, split_documents, get_vector_store, retrieve


class FakeLLM:
    """Runnable-compatible deterministic fake model for router parsing."""
    def __or__(self, other):
        return other

    def invoke(self, value):
        return AIMessage(content="summary")


def test_chunking() -> None:
    docs = load_text("Abstract\n\n" + ("Transformer models learn context. " * 100))
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=30)
    assert len(chunks) > 1
    assert all("page" in d.metadata for d in chunks)


def test_chroma_index_and_retrieval() -> None:
    text = "Short text classification uses contextual transformer representations. Attention pooling summarizes token features."
    docs = load_text(text)
    with tempfile.TemporaryDirectory() as tmp:
        store, _, chunks = get_vector_store(text.encode(), docs, chunk_size=300, persist_dir=tmp)
        assert chunks
        hits = retrieve(store, "transformer attention pooling", top_k=1)
        assert hits and "transformer" in hits[0][0].page_content.lower()


def test_router_parsing() -> None:
    # Exercise the router's label normalization using a minimal LCEL runnable.
    class StaticRunnable:
        def invoke(self, values):
            return "SUMMARY"
    router = IntentRouter.__new__(IntentRouter)
    router.chain = StaticRunnable()
    assert router.classify("summarize the paper") == "summary"


if __name__ == "__main__":
    test_chunking()
    test_chroma_index_and_retrieval()
    test_router_parsing()
    print("All offline smoke tests passed.")
