"""LCEL agents, intent routing, and agentic retrieval decisions."""
from __future__ import annotations
import json
import re
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import prompts
from rag import retrieve, format_hits


def _chain(llm: Any, system: str, user: str, parser: Any = None):
    p = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
    return p | llm | (parser or StrOutputParser())


def _json_fallback(text: str) -> dict:
    """Parse JSON safely, including a fenced JSON response."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {"raw": text}
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else {"raw": text}
            except Exception:
                pass
        return {"raw": text}


def _json_chain(llm: Any, system: str, user: str):
    return _chain(llm, system, user, JsonOutputParser())


class IntentRouter:
    def __init__(self, llm: Any):
        self.chain = _chain(llm, prompts.ROUTER_SYSTEM, prompts.ROUTER_USER)

    def classify(self, message: str, history: str = "") -> str:
        result = self.chain.invoke({"message": message, "history": history}).strip().lower()
        result = re.sub(r"[^a-z]", "", result)
        return result if result in {"summary", "slides", "concept", "viva", "chitchat"} else "chitchat"


class RetrievalDecisionAgent:
    def __init__(self, llm: Any):
        self.decision_chain = _json_chain(llm, prompts.RETRIEVAL_SYSTEM, prompts.RETRIEVAL_USER)
        self.sufficiency_chain = _json_chain(llm, prompts.SUFFICIENCY_SYSTEM, prompts.SUFFICIENCY_USER)

    def decide(self, message: str, history: str = "") -> dict:
        try:
            out = self.decision_chain.invoke({"message": message, "history": history})
            return {"decision": str(out.get("decision", "RETRIEVE")).upper(), "query": out.get("query", message), "reason": out.get("reason", "")}
        except Exception:
            raw = _chain  # preserve safe fallback without hiding paper requests
            meta = bool(re.search(r"\b(hello|hi|hey|thanks|thank you|what can you do|help)\b", message.lower()))
            return {"decision": "NO_RETRIEVE" if meta else "RETRIEVE", "query": message, "reason": "Safe local fallback"}

    def check(self, question: str, context: str) -> dict:
        try:
            out = self.sufficiency_chain.invoke({"question": question, "context": context[:9000]})
            return {"sufficient": bool(out.get("sufficient", True)), "query": out.get("query", ""), "reason": out.get("reason", "")}
        except Exception:
            return {"sufficient": True, "query": "", "reason": "Sufficiency check unavailable; using first retrieval."}


class SummarizerAgent:
    def __init__(self, llm: Any):
        self.draft = _chain(llm, prompts.SUMMARY_DRAFT_SYSTEM, prompts.SUMMARY_DRAFT_USER)
        self.reflect = _chain(llm, prompts.SUMMARY_CRITIQUE_SYSTEM, prompts.SUMMARY_CRITIQUE_USER)

    def run(self, context: str, request: str = "Summarize this paper", history: str = "") -> dict:
        draft = self.draft.invoke({"context": context[:14000], "request": request, "history": history})
        revised = self.reflect.invoke({"context": context[:14000], "draft": draft})
        return {"draft": draft, "summary": revised}


class ConceptExtractorAgent:
    def __init__(self, llm: Any):
        self.chain = _json_chain(llm, prompts.CONCEPT_SYSTEM, prompts.CONCEPT_USER)
        self.retry = _chain(llm, prompts.CONCEPT_SYSTEM, prompts.CONCEPT_USER)

    def run(self, context: str, history: str = "") -> dict:
        try:
            return self.chain.invoke({"context": context[:14000], "history": history})
        except Exception:
            raw = self.retry.invoke({"context": context[:14000], "history": history})
            return _json_fallback(raw)


class SlideGeneratorAgent:
    def __init__(self, llm: Any):
        self.chain = _json_chain(llm, prompts.SLIDES_SYSTEM, prompts.SLIDES_USER)
        self.retry = _chain(llm, prompts.SLIDES_SYSTEM, prompts.SLIDES_USER)

    def run(self, summary: str, concepts: str, history: str = "") -> dict:
        try:
            out = self.chain.invoke({"summary": summary[:9000], "concepts": concepts[:7000], "history": history})
        except Exception:
            out = _json_fallback(self.retry.invoke({"summary": summary[:9000], "concepts": concepts[:7000], "history": history}))
        slides = out.get("slides", []) if isinstance(out, dict) else []
        return {"slides": slides, "raw": out.get("raw", "") if isinstance(out, dict) else str(out)}


class VivaAgent:
    def __init__(self, llm: Any):
        self.chain = _json_chain(llm, prompts.VIVA_SYSTEM, prompts.VIVA_USER)
        self.retry = _chain(llm, prompts.VIVA_SYSTEM, prompts.VIVA_USER)
        self.practice = _json_chain(llm, prompts.PRACTICE_SYSTEM, prompts.PRACTICE_USER)
        self.practice_retry = _chain(llm, prompts.PRACTICE_SYSTEM, prompts.PRACTICE_USER)

    def run(self, context: str, summary: str, history: str = "") -> dict:
        values = {"context": context[:12000], "summary": summary[:5000], "history": history}
        try:
            out = self.chain.invoke(values)
        except Exception:
            out = _json_fallback(self.retry.invoke(values))
        return {"questions": out.get("questions", []), "raw": out.get("raw", "")}

    def score(self, question: str, model_answer: str, student_answer: str, context: str) -> dict:
        values = {"question": question, "model_answer": model_answer, "student_answer": student_answer, "context": context[:8000]}
        try:
            return self.practice.invoke(values)
        except Exception:
            return _json_fallback(self.practice_retry.invoke(values))


class PaperChatAgent:
    def __init__(self, llm: Any):
        self.chain = _chain(llm, prompts.CHAT_SYSTEM, prompts.CHAT_USER)

    def run(self, question: str, context: str, history: str = "") -> str:
        return self.chain.invoke({"question": question, "context": context[:12000], "history": history})


class AgentSuite:
    """Construct all task agents and dispatch an explicit task to its handler."""
    def __init__(self, llm: Any):
        self.router = IntentRouter(llm)
        self.retrieval = RetrievalDecisionAgent(llm)
        self.summarizer = SummarizerAgent(llm)
        self.concepts = ConceptExtractorAgent(llm)
        self.slides = SlideGeneratorAgent(llm)
        self.viva = VivaAgent(llm)
        self.chat = PaperChatAgent(llm)

    def route(self, intent: str, context: str, history: str = "", request: str = "") -> dict:
        if intent == "summary":
            return self.summarizer.run(context, request or "Summarize this paper", history)
        if intent == "concept":
            return self.concepts.run(context, history)
        if intent == "slides":
            return self.slides.run(request or "", context, history)
        if intent == "viva":
            return self.viva.run(context, request or "", history)
        return {"answer": self.chat.run(request, context, history)}
