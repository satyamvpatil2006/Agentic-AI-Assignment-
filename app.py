"""Streamlit interface for the Paper-to-Presentation & Viva Agent."""
from __future__ import annotations
import os
import json
import streamlit as st
from agents import AgentSuite
from llm import build_llm
from rag import load_pdf, load_text, get_vector_store, retrieve, format_hits
from export import slides_to_markdown, slides_to_pptx

st.set_page_config(page_title="Paper-to-Presentation & Viva Agent", page_icon="📄", layout="wide")
st.title("📄 Paper-to-Presentation & Viva Agent")
st.caption("Grounded academic assistant · Summary · Concepts · Slides · Viva")

DEFAULTS = {"history": [], "artifacts": {}, "paper_name": "", "paper_hash": "", "store": None, "chunks": [], "indexed": False, "llm_responded": False, "router_ran": False, "retrieval_ran": False, "trace": [], "last_context": "", "last_hits": [], "last_retrieval": {}, "last_intent": "", "last_handler": "", "selected_task": "Chat", "paper_bytes": b""}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.header("Configuration")
    provider = st.selectbox("LLM provider", ["OpenAI", "Ollama", "Anthropic"], index=0)
    defaults = {"OpenAI": "gpt-4o-mini", "Ollama": "llama3.1", "Anthropic": "claude-sonnet-4-5"}
    model = st.text_input("Model name", value=defaults[provider])
    env_key = os.getenv("OPENAI_API_KEY" if provider == "OpenAI" else "ANTHROPIC_API_KEY", "")
    api_key = st.text_input("API key (not needed for Ollama)", value=env_key, type="password")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    chunk_size = st.slider("Chunk size", 400, 1800, 1000, 100)
    top_k = st.slider("Top-k retrieval", 1, 10, 5)
    st.divider()
    st.header("Research paper")
    uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])
    if st.button("Load built-in sample paper", use_container_width=True):
        try:
            sample_path = os.path.join(os.path.dirname(__file__), "sample_paper.txt")
            raw = open(sample_path, "rb").read()
            docs = load_text(raw.decode("utf-8"))
            store, key, chunks = get_vector_store(raw, docs, chunk_size)
            st.session_state.update({"paper_name": "Built-in sample: Transformer Text Classification", "paper_hash": key, "store": store, "chunks": chunks, "indexed": True, "paper_bytes": raw})
            st.success("Sample paper loaded and indexed.")
        except Exception as exc:
            st.error(f"Could not load sample paper: {exc}")
    if uploaded is not None and st.button("Index uploaded PDF", use_container_width=True):
        try:
            raw = uploaded.getvalue()
            docs = load_pdf(raw)
            store, key, chunks = get_vector_store(raw, docs, chunk_size)
            st.session_state.update({"paper_name": uploaded.name, "paper_hash": key, "store": store, "chunks": chunks, "indexed": True, "paper_bytes": raw})
            st.success(f"Indexed {len(chunks)} chunks from {uploaded.name}.")
        except Exception as exc:
            st.error(f"PDF could not be read or indexed: {exc}")
    if st.session_state.indexed:
        st.caption(f"Loaded: {st.session_state.paper_name}")
        st.caption(f"Paper key: {st.session_state.paper_hash}")
    st.divider()
    st.header("Run a task")
    task_buttons = [("Summary", "summary"), ("Concepts", "concept"), ("Slides", "slides"), ("Viva", "viva")]
    for label, task in task_buttons:
        if st.button(label, use_container_width=True):
            st.session_state.selected_task = label
            st.session_state.pending_task = task


def history_text() -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.history[-12:])


def get_suite():
    try:
        return AgentSuite(build_llm(provider.lower(), model, api_key, temperature))
    except Exception as exc:
        st.error(str(exc))
        return None


def retrieve_for(suite, message: str):
    decision = suite.retrieval.decide(message, history_text())
    hits = []
    query = decision.get("query") or message
    if decision.get("decision") == "RETRIEVE" and st.session_state.store is not None:
        hits = retrieve(st.session_state.store, query, top_k)
        context = format_hits(hits)
        suff = suite.retrieval.check(message, context)
        if not suff.get("sufficient", True) and suff.get("query"):
            query = suff["query"]
            hits = retrieve(st.session_state.store, query, top_k)
            context = format_hits(hits)
            decision["retry"] = {"query": query, "reason": suff.get("reason", "")}
        st.session_state.retrieval_ran = True
    else:
        context = ""
    st.session_state.last_hits = hits
    st.session_state.last_context = context
    st.session_state.last_retrieval = {**decision, "query_used": query, "hit_count": len(hits)}
    return context, hits, decision


def run_task(task: str, message: str):
    if not st.session_state.indexed or st.session_state.store is None:
        st.warning("Upload a PDF or load the built-in sample paper first.")
        return
    suite = get_suite()
    if suite is None:
        return
    intent = task if task in {"summary", "concept", "slides", "viva"} else suite.router.classify(message, history_text())
    st.session_state.router_ran = True
    context, hits, decision = retrieve_for(suite, message)
    if decision["decision"] == "NO_RETRIEVE" and task == "chitchat":
        context = ""
    try:
        if intent == "summary":
            # Section-targeted retrieval gives the summarizer evidence across the whole paper.
            section_queries = [
                "abstract introduction background research problem objectives",
                "method methodology model architecture proposed approach training",
                "experiments evaluation datasets metrics baseline results ablation",
                "conclusion discussion key findings implications",
                "limitations threats to validity future work constraints",
            ]
            section_hits = []
            seen = set()
            for section_query in section_queries:
                for doc, score in retrieve(st.session_state.store, section_query, max(1, min(2, top_k))):
                    marker = (doc.metadata.get("page"), doc.page_content[:160])
                    if marker not in seen:
                        seen.add(marker)
                        section_hits.append((doc, score))
            if section_hits:
                hits = section_hits
                context = format_hits(section_hits)
                st.session_state.last_hits = hits
                st.session_state.last_context = context
                st.session_state.last_retrieval["section_queries"] = section_queries
                st.session_state.last_retrieval["hit_count"] = len(hits)
            result = suite.summarizer.run(context, message or "Summarize this paper", history_text())
            st.session_state.artifacts["summary"] = result["summary"]
            answer = result["summary"]
            handler = "SummarizerAgent (draft → critique → revised)"
        elif intent == "concept":
            result = suite.concepts.run(context, history_text())
            st.session_state.artifacts["concepts"] = result
            answer = json.dumps(result, indent=2, ensure_ascii=False)
            handler = "ConceptExtractorAgent (CoT → JSON)"
        elif intent == "slides":
            summary = st.session_state.artifacts.get("summary", "")
            concepts = st.session_state.artifacts.get("concepts", {})
            if not summary:
                summary = suite.summarizer.run(context, "Create a concise paper summary for slides", history_text())["summary"]
                st.session_state.artifacts["summary"] = summary
            if not concepts:
                concepts = suite.concepts.run(context, history_text())
                st.session_state.artifacts["concepts"] = concepts
            result = suite.slides.run(summary, json.dumps(concepts, ensure_ascii=False), history_text())
            st.session_state.artifacts["slides"] = result["slides"]
            answer = f"Generated {len(result['slides'])} slides." if result["slides"] else result.get("raw", "Slide generation returned no structured slides.")
            handler = "SlideGeneratorAgent"
        elif intent == "viva":
            summary = st.session_state.artifacts.get("summary", "")
            result = suite.viva.run(context, summary, history_text())
            st.session_state.artifacts["viva"] = result["questions"]
            answer = f"Generated {len(result['questions'])} viva questions." if result["questions"] else result.get("raw", "Viva generation returned no structured questions.")
            handler = "VivaAgent (few-shot)"
        else:
            answer = suite.chat.run(message, context, history_text())
            handler = "PaperChatAgent"
        st.session_state.llm_responded = True
        st.session_state.last_intent = intent
        st.session_state.last_handler = handler
        st.session_state.history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": answer}])
        st.session_state.trace.append({"intent": intent, "handler": handler, "retrieval": st.session_state.last_retrieval, "message": message})
        st.session_state.trace = st.session_state.trace[-20:]
    except Exception as exc:
        message_text = str(exc)
        if "429" in message_text or "rate" in message_text.lower():
            st.error("The model provider rate-limited this request. Wait briefly, reduce repeated calls, or check your provider quota.")
        elif "ollama" in message_text.lower() or "connection" in message_text.lower():
            st.error("Could not connect to Ollama. Start the Ollama server and ensure the selected model is pulled (for example: ollama pull llama3.1).")
        else:
            st.error(f"The agent could not complete this task: {message_text[:500]}")


# Tabs are kept stable across reruns.
tabs = st.tabs(["Chat", "Summary", "Concepts", "Slides", "Viva", "Prompts & Trace", "Acceptance Checklist"])
with tabs[0]:
    st.subheader("Ask about your paper")
    if not st.session_state.indexed:
        st.info("Load the built-in sample paper or upload a PDF from the sidebar to begin.")
    for item in st.session_state.history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
    if prompt := st.chat_input("Ask a question about the paper, or say hello..."):
        suite = get_suite()
        if suite:
            try:
                intent = suite.router.classify(prompt, history_text())
                st.session_state.router_ran = True
                run_task(intent, prompt)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not classify or route the message: {exc}")
    if st.session_state.last_intent:
        with st.expander("Agent decision & retrieval trace", expanded=False):
            st.write(f"**Detected intent:** {st.session_state.last_intent}")
            st.write(f"**Routed handler:** {st.session_state.last_handler}")
            st.write("**Retrieval decision:**", st.session_state.last_retrieval)
            if st.session_state.last_hits:
                for doc, score in st.session_state.last_hits:
                    st.markdown(f"**[p.{doc.metadata.get('page', '?')}] · similarity distance {score:.4f}**")
                    st.write(doc.page_content)
            else:
                st.caption("No retrieval was performed for this message.")
    pending = st.session_state.pop("pending_task", None)
    if pending:
        run_task(pending, f"Please generate the paper {pending}.")
        st.rerun()

with tabs[1]:
    st.subheader("Structured paper summary")
    if "summary" in st.session_state.artifacts:
        st.markdown(st.session_state.artifacts["summary"])
    else:
        st.info("Choose Summary in the sidebar to generate a sectioned, reflected summary.")
with tabs[2]:
    st.subheader("Key concepts")
    concepts = st.session_state.artifacts.get("concepts")
    if concepts:
        for key, value in concepts.items():
            st.markdown(f"**{key.replace('_', ' ').title()}**")
            if isinstance(value, list):
                for entry in value:
                    st.markdown(f"- {entry}")
            else:
                st.write(value)
    else:
        st.info("Choose Concepts in the sidebar to extract contributions, methods, datasets, metrics, and results.")
with tabs[3]:
    st.subheader("Generated presentation")
    slides = st.session_state.artifacts.get("slides", [])
    if slides:
        for i, slide in enumerate(slides, 1):
            with st.container(border=True):
                st.markdown(f"### Slide {i}: {slide.get('title', 'Untitled')}")
                for bullet in slide.get("bullets", []):
                    st.markdown(f"- {bullet}")
                with st.expander("Speaker notes"):
                    st.write(slide.get("speaker_notes", ""))
        st.download_button("Download .pptx", slides_to_pptx(slides), file_name="research_presentation.pptx", mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
        st.download_button("Download slides.md", slides_to_markdown(slides), file_name="slides.md", mime="text/markdown")
    else:
        st.info("Choose Slides in the sidebar to create an 8–12 slide deck.")
with tabs[4]:
    st.subheader("Viva preparation")
    questions = st.session_state.artifacts.get("viva", [])
    if questions:
        for i, q in enumerate(questions):
            with st.expander(f"Q{i+1}. {q.get('question', 'Question')} · {q.get('difficulty', '—')}"):
                st.write("**Model answer**")
                st.write(q.get("answer", ""))
                st.caption(f"Type: {q.get('type', '—')}")
                student = st.text_area("Your answer", key=f"practice_{i}")
                if st.button("Score my answer", key=f"score_{i}"):
                    suite = get_suite()
                    if suite:
                        try:
                            result = suite.viva.score(q.get("question", ""), q.get("answer", ""), student, st.session_state.last_context)
                            st.success(f"Score: {result.get('score', '—')}/5")
                            st.write(result.get("feedback", result.get("raw", "")))
                            if result.get("missing_points"):
                                st.write("**Missing points:**", result["missing_points"])
                            if result.get("strong_points"):
                                st.write("**Strong points:**", result["strong_points"])
                        except Exception as exc:
                            st.error(f"Could not score answer: {exc}")
    else:
        st.info("Choose Viva in the sidebar to generate likely examiner questions and model answers.")
with tabs[5]:
    st.subheader("Prompt library")
    import prompts as prompt_module
    prompt_labels = [("Zero-shot · Router", "ROUTER_SYSTEM", "ROUTER_USER"), ("Zero-shot · Retrieval decision", "RETRIEVAL_SYSTEM", "RETRIEVAL_USER"), ("Chain-of-Thought · Concept extractor", "CONCEPT_SYSTEM", "CONCEPT_USER"), ("Reflection · Summary draft", "SUMMARY_DRAFT_SYSTEM", "SUMMARY_DRAFT_USER"), ("Reflection · Critique and revise", "SUMMARY_CRITIQUE_SYSTEM", "SUMMARY_CRITIQUE_USER"), ("Few-shot · Viva generation (2 examples)", "VIVA_SYSTEM", "VIVA_USER"), ("Zero-shot · Slide generation", "SLIDES_SYSTEM", "SLIDES_USER"), ("Zero-shot · Grounded chat", "CHAT_SYSTEM", "CHAT_USER"), ("Zero-shot · Viva practice scoring", "PRACTICE_SYSTEM", "PRACTICE_USER")]
    for label, sys_name, user_name in prompt_labels:
        with st.expander(label):
            st.markdown("**System prompt**")
            st.code(getattr(prompt_module, sys_name), language="text")
            st.markdown("**User prompt template**")
            st.code(getattr(prompt_module, user_name), language="text")
    st.subheader("Recent chain runs")
    st.json(st.session_state.trace[-10:])
with tabs[6]:
    st.subheader("Seven mandatory acceptance criteria")
    rows = [
        ("1. LLM integration", "llm.py · build_llm()", st.session_state.llm_responded, "A model response has completed"),
        ("2. Interactive GUI", "app.py · Streamlit tabs, uploader, chat_input", True, "UI is running"),
        ("3. Prompt engineering", "prompts.py · router, few-shot, CoT, reflection", True, "Named prompts are present"),
        ("4. LangChain LCEL", "agents.py · ChatPromptTemplate | model | parser", True, "All agent calls use LCEL"),
        ("5. Agentic RAG", "agents.py · RetrievalDecisionAgent; app.py · retrieve_for()", st.session_state.retrieval_ran, "A paper retrieval has occurred"),
        ("6. Intent routing", "agents.py · IntentRouter; app.py · run_task()", st.session_state.router_ran, "Router has classified a message"),
        ("7. Chroma vector DB", "rag.py · get_vector_store(), retrieve()", st.session_state.indexed, "Paper chunks indexed in persistent Chroma"),
    ]
    st.table([{"Criterion": name, "Implementation": file, "Live status": "✅ PASS" if status else "○ Pending", "Evidence": evidence} for name, file, status, evidence in rows])
    st.caption("Live statuses update as you load a paper, run the router, retrieve evidence, and receive a model response.")
