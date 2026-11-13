# Paper-to-Presentation & Viva Agent

A Streamlit academic assistant that indexes an uploaded research-paper PDF in persistent ChromaDB, routes requests to LangChain LCEL agents, and generates a grounded summary, concept extraction, slide deck, and viva practice questions. The local Chroma `DefaultEmbeddingFunction` supplies ONNX MiniLM embeddings without a separate embedding API. The app supports OpenAI, local Ollama, and Anthropic chat models.

**Assumption:** PDFs are text-based; scanned image-only PDFs are rejected with a readable message rather than silently OCR'd. The bundled sample is plain text so the demo works without downloading a paper.

## Setup in 3 commands

```bash
cd paper_viva_agent
python -m venv .venv
```

```bash
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

Choose a provider in the sidebar. For OpenAI or Anthropic, paste the key into the password field or set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` in the environment. Ollama requires a running local server and a pulled model, e.g. `ollama pull llama3.1`. No `.env` file is needed.

## How to use

1. Load the built-in sample paper or upload a text-based PDF, then click **Index uploaded PDF**.
2. Choose Summary, Concepts, Slides, or Viva in the sidebar, or ask a question in Chat.
3. Inspect the per-message trace expander for intent, handler, retrieval decision, query, excerpts, and distance scores.
4. In Slides, download `research_presentation.pptx` or `slides.md`. In Viva, enter an answer and score it against the model answer.
5. Review **Prompts & Trace** and **Acceptance Checklist** for prompt techniques and live evidence.

## Architecture

```text
User
  |
  v
Intent Router (zero-shot) --------------------> Chitchat (NO_RETRIEVE)
  |
  v
Retrieval Decision (RETRIEVE / NO_RETRIEVE + query rewrite)
  |
  v
ChromaDB (paper-hash collection; local MiniLM; top-k similarity)
  |                         ^
  |-- sufficiency check ----| (one reformulated retry if needed)
  v
Agent chain (summary / concepts / slides / viva / grounded chat)
  |
  v
Streamlit UI + trace + downloads + session-state history
```

Every LLM call is a LangChain LCEL pipeline using `ChatPromptTemplate | model | parser`. Prompt templates are named in `prompts.py`. The summary uses a draft → critique/revise reflection pass; concepts use a Chain-of-Thought-labelled prompt that requests private stepwise analysis but only returns final JSON; viva generation includes two worked few-shot examples. JSON chains fall back to a text parser if structured parsing fails.

## Acceptance screenshot checklist

| Criterion | Where to demonstrate / screenshot | Evidence |
|---|---|---|
| 1. LLM integration | Chat tab after asking a question; Acceptance Checklist | Model response and live PASS |
| 2. Interactive GUI | Main app with sidebar uploader and Chat tab | PDF upload, task buttons, chat input, tabs |
| 3. Prompt engineering | Prompts & Trace tab | Zero-shot router, few-shot viva (two examples), CoT concept, reflection prompts |
| 4. LangChain core | Prompts & Trace plus source `agents.py` | LCEL chain construction with `ChatPromptTemplate`, model, and parsers |
| 5. Agentic RAG | Chat trace expander after a paper question | Retrieval decision, rewritten query, retrieved chunks and scores; retry metadata if needed |
| 6. Intent routing | Chat trace expander | Detected intent and routed handler for the message |
| 7. Vector DB | Sidebar after indexing and Acceptance Checklist | Paper key, indexed status; Chroma retrieval evidence in trace |

## Five-prompt demo script

1. `Hi! What can you do?` — demonstrates chitchat and NO_RETRIEVE.
2. `Summarize the research problem, method, results, and limitations.` — demonstrates summary reflection and sectioned output.
3. `What are the paper's main contributions, datasets, metrics, and findings?` — demonstrates concept extraction and structured JSON.
4. `Create a 10-slide presentation for a 5-minute class presentation.` — demonstrates slide generation and exports.
5. `Generate likely viva questions, then ask me a hard question about the limitations.` — demonstrates few-shot viva generation; use Practice mode to score an answer.

## Tests

Run the offline smoke tests with:

```bash
python test_smoke.py
```

The test checks chunking, local Chroma indexing/retrieval, and router label parsing. It does not require a chat-model API key. Chroma's first embedding run may download its bundled ONNX model once; after that it is cached locally.

## Notes

- Each paper gets a persistent collection named from a SHA-256 file hash in `./chroma_db`.
- Similarity scores from Chroma are distances (lower is closer), not probabilities.
- The app sends only retrieved excerpts and bounded conversation history to the selected LLM. Outputs remain subject to the paper's coverage and quality.
- If the paper does not state a fact, agents are instructed to say so rather than invent it.
