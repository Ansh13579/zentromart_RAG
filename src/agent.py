"""Initial LLM-only ZentroMart support agent.

The ``respond`` method is the Step 1 baseline. ``respond_with_rag`` adds the
Step 2 retrieval and grounded-generation flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import json
from pathlib import Path
import re
from typing import Any, Protocol

if __package__:
    from .llm_client import LLMClient
    from .prompts import (
        format_conversation_history,
        format_retrieved_context,
        get_final_conversation_prompt,
        get_grounded_rag_system_prompt,
        get_initial_system_prompt,
        get_tool_aware_system_prompt,
    )
    from .data_store import DataStore
    from .policy_facts import add_calendar_evidence, calculate_policy_facts
    from .retriever import DEFAULT_INDEX_DIR, Retriever
else:
    from llm_client import LLMClient
    from prompts import (
        format_conversation_history,
        format_retrieved_context,
        get_final_conversation_prompt,
        get_grounded_rag_system_prompt,
        get_initial_system_prompt,
        get_tool_aware_system_prompt,
    )
    from data_store import DataStore
    from policy_facts import add_calendar_evidence, calculate_policy_facts
    from retriever import DEFAULT_INDEX_DIR, Retriever


class RetrieverLike(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return ranked evidence chunks for a query."""


class OrderLookup(Protocol):
    def __call__(
        self,
        order_id: str,
        verification_value: str | None = None,
    ) -> dict[str, Any]:
        """Return identity-gated order information."""


@dataclass(frozen=True)
class ToolAwareResponse:
    """Answer plus evidence, tool trace and orchestration metadata.

    ``answer`` remains the compatibility name used by the conversation layer.
    The ``reply`` property exposes the Phase 4 result shape without duplicating
    the generated response.
    """

    answer: str
    sources: list[dict[str, Any]]
    retrieval_scores: list[float]
    tool_results: dict[str, Any]
    abstained: bool
    tool_calls: list[str] = field(default_factory=list)
    ticket_id: str | None = None
    state_changes: dict[str, Any] = field(default_factory=dict)
    needs_human: bool = False

    @property
    def reply(self) -> str:
        """Return the generated customer-facing reply."""

        return self.answer


@dataclass(frozen=True)
class RAGResponse:
    """Answer and retrieval trace returned by the RAG agent."""

    answer: str
    sources: list[dict[str, Any]]
    retrieval_scores: list[float]
    abstained: bool
    retrieved_sources: list[dict[str, Any]] = field(default_factory=list)


class SupportAgent:
    """Generate Step 1 and Step 2 support responses."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        retriever: RetrieverLike | None = None,
        *,
        index_dir: Path = DEFAULT_INDEX_DIR,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.system_prompt = get_initial_system_prompt()
        self.rag_system_prompt = get_grounded_rag_system_prompt()
        self.tool_system_prompt = get_tool_aware_system_prompt()
        self.retriever = retriever
        self.index_dir = index_dir

    def respond(self, customer_question: str) -> str:
        """Return an LLM response to one customer question."""

        question = customer_question.strip()
        if not question:
            return "Please tell me how I can help with your ZentroMart request."

        response = self.llm_client.generate(
            system_message=self.system_prompt,
            user_message=question,
        )

        if response is None or not response.strip():
            raise RuntimeError(
                "The language model did not return a response."
            )

        return response.strip()

    def respond_with_rag(
        self,
        customer_question: str,
        *,
        top_k: int = 5,
    ) -> RAGResponse:
        """Retrieve evidence, generate a grounded answer and return its trace."""

        question = customer_question.strip()
        if not question:
            raise ValueError("customer_question must not be empty.")

        retriever = self._get_retriever()
        retrieved_chunks = retriever.search(question, top_k=top_k)
        ranked_chunks = list(retrieved_chunks)
        retrieved_chunks = add_calendar_evidence(question, retrieved_chunks)
        calendar_facts = calculate_policy_facts(question, retrieved_chunks, DataStore())
        context = format_retrieved_context(retrieved_chunks)
        user_message = (
            f"Customer question:\n{question}\n\n"
            f"Retrieved evidence:\n{context}\n\n"
            f"Calculated policy facts (use exact ISO dates):\n{json.dumps(calendar_facts, ensure_ascii=False)}"
        )
        answer = self.llm_client.generate(
            system_message=self.rag_system_prompt,
            user_message=user_message,
        )
        if answer is None or not answer.strip():
            raise RuntimeError("The language model did not return a RAG answer.")

        sources = self._export_sources(retrieved_chunks, answer)
        scores = [
            float(chunk.get("rank_score", chunk.get("hybrid_score", chunk["score"])))
            for chunk in ranked_chunks
        ]
        return RAGResponse(
            answer=answer.strip(),
            sources=sources,
            retrieval_scores=scores,
            abstained=self._is_abstention(answer),
            retrieved_sources=self._export_sources(ranked_chunks, answer),
        )

    def respond_with_tools(
        self,
        customer_question: str,
        *,
        verification_value: str | None = None,
        verified_order: dict[str, Any] | None = None,
        top_k: int = 5,
        data_store: DataStore | None = None,
        order_lookup: OrderLookup | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        customer_language: str = "en",
        current_topic: str = "general",
    ) -> ToolAwareResponse:
        """Answer using policy retrieval plus trusted structured-data tools.

        Public product and pincode facts come from ``DataStore``.  Order
        details are always obtained through the identity-gated order tool.
        ``verification_value`` is caller-supplied and is never included in
        the prompt or returned trace.
        """

        question = customer_question.strip()
        if not question:
            raise ValueError("customer_question must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        retriever = self._get_retriever()
        # Resolve short follow-ups with recent customer context for retrieval.
        prior = [
            turn["content"] for turn in (conversation_history or [])
            if turn.get("role") == "customer" and turn.get("content") != question
        ]
        retrieval_query = question
        if current_topic not in {"", "general", "verification"}:
            retrieval_query += f"\nTopic: {current_topic}"
        if len(question.split()) < 12 and prior:
            retrieval_query += f"\nPrevious customer request: {prior[-1]}"
        retrieved_chunks = retriever.search(retrieval_query, top_k=top_k)
        retrieved_chunks = add_calendar_evidence(
            question + f"\nTopic: {current_topic}", retrieved_chunks,
            verified_order=verified_order,
        )
        store = data_store or DataStore()
        tool_results = self._collect_tool_results(
            question,
            store=store,
            order_lookup=order_lookup,
            verification_value=verification_value,
            verified_order=verified_order,
        )
        calendar_facts = calculate_policy_facts(question, retrieved_chunks, store, verified_order)
        if calendar_facts:
            tool_results["calculated_policy_facts"] = calendar_facts
        context = format_retrieved_context(retrieved_chunks)
        history = format_conversation_history(conversation_history or [])
        user_message = (
            f"Customer language: {customer_language}\n"
            f"Current topic: {current_topic}\n\n"
            f"Conversation history:\n{history}\n\n"
            f"Current customer question:\n{question}\n\n"
            "Dataset date rules:\n"
            "- Today is 20 October 2026.\n"
            "- Business days are Monday through Saturday, excluding holidays.\n"
            "- Return windows use calendar days.\n\n"
            f"Retrieved policy evidence:\n{context}\n\n"
            f"Trusted application/tool results:\n{self._safe_tool_context(tool_results)}"
        )
        answer = self.llm_client.generate(
            system_message=get_final_conversation_prompt(),
            user_message=user_message,
        )
        if answer is None or not answer.strip():
            raise RuntimeError(
                "The language model did not return a tool-aware answer."
            )

        sources = self._export_sources(retrieved_chunks, answer)
        return ToolAwareResponse(
            answer=answer.strip(),
            sources=sources,
            retrieval_scores=[
                float(chunk.get("rank_score", chunk.get("hybrid_score", chunk["score"])))
                for chunk in retrieved_chunks
            ],
            tool_results=tool_results,
            abstained=self._is_abstention(answer),
            tool_calls=self._tool_call_names(tool_results),
            state_changes=self._state_changes(verified_order),
            needs_human=self._needs_human(answer, tool_results),
        )

    @staticmethod
    def _export_sources(chunks: list[dict[str, Any]], answer: str) -> list[dict[str, Any]]:
        """Save full evidence and mark only citations actually emitted by the LLM."""

        cited = set()
        for match in re.finditer(r"(?:Sources?|स्रोत)\s*:\s*([^\n]+)", answer, re.IGNORECASE):
            cited.update(re.findall(r"[A-Z][A-Z0-9]*-\d+(?:\.\d+)?", match[1]))
        return [
            {
                **chunk,
                "evidence_role": chunk.get("evidence_role", "retrieved"),
                "cited_in_answer": chunk.get("section_id") in cited,
            }
            for chunk in chunks
        ]

    @staticmethod
    def _collect_tool_results(
        question: str,
        *,
        store: DataStore,
        order_lookup: OrderLookup | None,
        verification_value: str | None,
        verified_order: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Collect only data relevant to IDs explicitly present in a question."""

        results: dict[str, Any] = {}
        sku_matches = re.findall(r"\b[A-Z]{2,5}-\d{4}\b", question.upper())
        if sku_matches:
            results["products"] = [
                product
                for sku in dict.fromkeys(sku_matches)
                for product in store.find_product(sku=sku)
            ]

        pincode_matches = re.findall(r"\b\d{6}\b", question)
        if pincode_matches:
            results["pincodes"] = [
                {"pincode": pincode, "record": store.check_pincode(pincode)}
                for pincode in dict.fromkeys(pincode_matches)
            ]

        order_matches = re.findall(r"\b\d{8}\b", question)
        if verified_order is not None:
            results["orders"] = [
                SupportAgent._redact_order_tool_result(verified_order)
            ]
        elif order_matches:
            lookup = order_lookup or SupportAgent._load_default_order_lookup()
            results["orders"] = []
            for order_id in dict.fromkeys(order_matches):
                result = lookup(order_id, verification_value=verification_value)
                results["orders"].append(
                    SupportAgent._redact_order_tool_result(result)
                )
        return results

    @staticmethod
    def _redact_order_tool_result(result: dict[str, Any]) -> dict[str, Any]:
        """Keep verification failures safe and remove accidental secrets."""

        if not result.get("verified"):
            return {
                key: value
                for key, value in result.items()
                if key in {"verification_required", "verified", "message", "order_id"}
            }
        return {
            key: value
            for key, value in result.items()
            if key not in {"phone", "verification_value"}
        }

    @staticmethod
    def _safe_tool_context(tool_results: dict[str, Any]) -> str:
        """Serialize tool results without exposing an untrusted instruction."""

        if not tool_results:
            return "No structured-data tool was needed for this question."
        return repr(tool_results)

    @staticmethod
    def _tool_call_names(tool_results: dict[str, Any]) -> list[str]:
        """Map collected result groups to stable application tool names."""

        calls: list[str] = []
        if "products" in tool_results:
            calls.append("find_product")
        if "pincodes" in tool_results:
            calls.append("check_pincode")
        if "orders" in tool_results:
            calls.append("lookup_order")
        return calls

    @staticmethod
    def _state_changes(
        verified_order: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Expose only safe orchestration changes made for this request."""

        if verified_order and verified_order.get("verified"):
            return {"verification_status": "verified"}
        return {}

    @staticmethod
    def _needs_human(answer: str, tool_results: dict[str, Any]) -> bool:
        """Flag replies that explicitly require human review or escalation."""

        text = answer.casefold()
        handover_terms = (
            "human support",
            "human colleague",
            "support colleague",
            "human agent",
            "handed this over",
            "contact a support",
            "मानव सहयोगी",
            "सहयोगी से",
            "मानव एजेंट",
            "मानव सहायता",
        )
        return any(term in text for term in handover_terms) or (
            not tool_results and SupportAgent._is_abstention(answer)
        )

    @staticmethod
    def _load_default_order_lookup() -> OrderLookup:
        """Load the supplied offline identity-gated order tool."""

        tools_path = (
            Path(__file__).resolve().parents[2]
            / "Capstone_Project_Dataset_FTTT_2026"
            / "tools"
            / "tools.py"
        )
        if not tools_path.is_file():
            raise FileNotFoundError(f"Offline order tool is missing: {tools_path}")
        spec = importlib.util.spec_from_file_location(
            "zentromart_offline_tools", tools_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load offline order tool: {tools_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.lookup_order

    def _get_retriever(self) -> RetrieverLike:
        if self.retriever is None:
            loaded_retriever = Retriever()
            loaded_retriever.load(self.index_dir)
            self.retriever = loaded_retriever
        return self.retriever

    @staticmethod
    def _is_abstention(answer: str) -> bool:
        # LLMs commonly emit typographic apostrophes. Normalize them before
        # checking the refusal language so a correct abstention is not counted
        # as an answer merely because it used a curly quote.
        text = (
            answer.casefold()
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u02bc", "'")
        )
        abstention_phrases = (
            "i don't have enough information",
            "i do not have enough information",
            "don't have enough information",
            "do not have enough information",
            "i can't confirm",
            "i cannot confirm",
            "can't confirm",
            "cannot confirm",
            "i don't know",
            "i do not know",
            "not enough information",
            "जानकारी नहीं",
            "pata nahi",
            "jaankari nahi",
        )
        return any(phrase in text for phrase in abstention_phrases)


if __name__ == "__main__":
    agent = SupportAgent()
    question = input("Customer: ")
    print(f"Assistant: {agent.respond(question)}")
