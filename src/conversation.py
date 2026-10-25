"""Stateful, identity-aware conversation handling for ZentroMart support."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import importlib.util
from pathlib import Path
import re
from typing import Any, Callable, Protocol

if __package__:
    from .agent import OrderLookup, SupportAgent, ToolAwareResponse
else:
    from agent import OrderLookup, SupportAgent, ToolAwareResponse


class TicketCreator(Protocol):
    def __call__(
        self,
        summary: str,
        order_id: str | None = None,
        reason: str | None = None,
        intent: str | None = None,
    ) -> dict[str, Any]:
        """Create a human-handover ticket."""


@dataclass
class ConversationState:
    """Mutable state for one conversation only."""

    current_order_id: str | None = None
    verification_status: str = "not_required"
    verification_attempts: int = 0
    current_issue: str = ""
    current_topic: str = ""
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    customer_language: str = "en"
    escalation_state: str = "none"
    ticket_id: str | None = None
    verified_order: dict[str, Any] | None = None
    introduced: bool = False

    def public_snapshot(self) -> dict[str, Any]:
        """Return state safe to display or persist outside the session."""

        snapshot = asdict(self)
        snapshot.pop("verified_order", None)
        return snapshot


@dataclass(frozen=True)
class ConversationResponse:
    """One assistant turn plus a safe state snapshot."""

    answer: str
    state: dict[str, Any]
    tool_response: ToolAwareResponse | None = None


class ConversationSession:
    """Manage order verification and tool-aware answers across turns."""

    def __init__(
        self,
        *,
        agent: SupportAgent | None = None,
        order_lookup: OrderLookup | None = None,
        ticket_creator: TicketCreator | None = None,
        state: ConversationState | None = None,
    ) -> None:
        self.agent = agent or SupportAgent()
        self.order_lookup = order_lookup or SupportAgent._load_default_order_lookup()
        self.ticket_creator = ticket_creator or self._load_default_ticket_creator()
        self.state = state or ConversationState()
        self._verification_secret: str | None = None
        self._order_attempts: dict[str, int] = {}
        self._turn_calls: list[str] = []
        self._turn_results: dict[str, Any] = {}

    def process_message(self, customer_message: str) -> ConversationResponse:
        """Process one customer turn without exposing verification secrets."""

        message = customer_message.strip()
        if not message:
            raise ValueError("customer_message must not be empty.")
        self._turn_calls = []
        self._turn_results = {}
        self.state.customer_language = self._detect_language(
            message, self.state.customer_language
        )
        detected_topic = self._detect_topic(message)
        if detected_topic not in {"general", "verification"} and self.state.verification_status != "pending":
            self.state.current_topic = detected_topic

        order_id = self._extract_order_id(message)
        if order_id and order_id != self.state.current_order_id:
            # Previous order details must not enter a new order's prompt.
            self.state.conversation_history.clear()
            self.state.current_order_id = order_id
            self.state.current_topic = detected_topic
            self.state.verified_order = None
            self.state.ticket_id = None
            self.state.escalation_state = "none"
            self._verification_secret = None
            self.state.verification_status = "pending"
            self.state.verification_attempts = self._order_attempts.get(order_id, 0)
            self._record("customer", message)
            self.state.current_issue = message
            if self.state.verification_attempts >= 2:
                self.state.verification_status = "failed"
                return self._response(self._verification_failure())
            pending = self.order_lookup(order_id)
            self._turn_calls.append("lookup_order")
            self._turn_results["lookup_order"] = SupportAgent._redact_order_tool_result(pending)
            if not pending.get("verification_required", False):
                raise RuntimeError(
                    "The order tool did not enforce verification for an unverified lookup."
                )
            answer = self._verification_request()
            return self._response(answer)

        self._record("customer", message)
        if self.state.verification_status == "pending":
            verification_value = self._extract_verification_value(message, order_id)
            if verification_value is None:
                answer = self._verification_request()
                return self._response(answer)
            return self._attempt_verification(verification_value, message)

        if self.state.verification_status == "failed":
            answer = self._verification_failure()
            return self._response(answer)

        self.state.current_issue = message
        if self._should_escalate(message):
            return self.escalate(
                message,
                reason=self._escalation_reason(message),
            )

        if self.state.current_order_id is None and self._needs_order_id(message):
            answer = (
                "Please share your "
                "eight-digit order ID so I can help."
            )
            return self._response(answer)

        verified_order = (
            self.state.verified_order
            if self.state.verification_status == "verified"
            else None
        )
        order_context = message
        if verified_order is not None and self.state.current_order_id:
            order_context = f"{message}\nOrder ID: {self.state.current_order_id}"
        result = self.agent.respond_with_tools(
            order_context,
            verification_value=None,
            verified_order=verified_order,
            conversation_history=self.state.conversation_history,
            customer_language=self.state.customer_language,
            current_topic=self.state.current_topic,
        )
        return self._finish_agent_response(result, message)

    def _finish_agent_response(
        self, result: ToolAwareResponse, message: str,
    ) -> ConversationResponse:
        """Convert a generated handover recommendation into a confirmed action."""

        if result.needs_human and not result.abstained:
            handover = self.escalate(message, reason="policy_requires_human_review")
            return replace(
                handover,
                tool_response=replace(
                    handover.tool_response,
                    sources=result.sources,
                    retrieval_scores=result.retrieval_scores,
                    tool_calls=list(dict.fromkeys(
                        [call for call in result.tool_calls if call != "lookup_order"]
                        + handover.tool_response.tool_calls
                    )),
                    tool_results={**result.tool_results, **handover.tool_response.tool_results},
                ),
            )
        return self._response(result.answer, result)

    def escalate(
        self,
        summary: str,
        *,
        reason: str,
        intent: str | None = None,
    ) -> ConversationResponse:
        """Create a handover only when the ticket tool confirms creation."""

        if not summary.strip():
            raise ValueError("summary must not be empty.")
        handover_summary = self._redact_secret(
            f"Issue: {summary.strip()}; Intent: {intent or self.state.current_topic or 'general'}; "
            f"Order: {self.state.current_order_id or 'not supplied'}; "
            f"Verification: {self.state.verification_status}; "
            f"Checks: identity-gated lookup {'completed' if self.state.verified_order else 'not completed'}; "
            f"Customer context: {self.state.current_issue}; Reason: {reason}."
        )
        result = self.ticket_creator(
            summary=handover_summary,
            order_id=self.state.current_order_id,
            reason=reason,
            intent=intent,
        )
        if not result.get("created"):
            raise RuntimeError(
                f"Ticket creation failed: {result.get('error', 'unknown error')}"
            )
        ticket_id = str(result.get("ticket_id", "")).strip()
        if not ticket_id:
            raise RuntimeError("Ticket tool reported success without a ticket ID.")
        self.state.escalation_state = "escalated"
        self.state.ticket_id = ticket_id
        reference_number = self._verified_reference_number()
        answer = (
            f"I created ticket {ticket_id} and handed this over to our support "
            "team for review."
        )
        if reference_number:
            answer += f" Your refund reference number (RRN) is {reference_number}."
        self._turn_calls.append("create_ticket")
        self._turn_results["create_ticket"] = {
            **result, "summary": handover_summary, "reason": reason,
        }
        return self._response(answer)

    def _attempt_verification(
        self,
        verification_value: str,
        message: str,
    ) -> ConversationResponse:
        self.state.verification_attempts += 1
        self._order_attempts[self.state.current_order_id or ""] = self.state.verification_attempts
        self._verification_secret = verification_value
        result = self.order_lookup(
            self.state.current_order_id or "",
            verification_value=verification_value,
        )
        self._turn_calls.append("lookup_order")
        self._turn_results["lookup_order"] = SupportAgent._redact_order_tool_result(result)
        if result.get("verified") and result.get("found"):
            self.state.verification_status = "verified"
            self.state.verified_order = result
            order_context = (
                f"{self.state.current_issue}\n"
                f"Order ID: {self.state.current_order_id}"
                "\nVerification succeeded. Give a concise order-specific answer "
                "or summary first; do not ask the customer to repeat the order "
                "ID or verification value, and do not ask a generic 'what help "
                "do you need?' clarification."
            )
            answer_result = self.agent.respond_with_tools(
                order_context,
                verification_value=verification_value,
                verified_order=result,
                conversation_history=self.state.conversation_history,
                customer_language=self.state.customer_language,
                current_topic=self.state.current_topic,
            )
            return self._finish_agent_response(answer_result, self.state.current_issue)

        if self.state.verification_attempts >= 2:
            self.state.verification_status = "failed"
            answer = (
                "I couldn't verify the order after two attempts, so I can't "
                "share order details. Please contact a support colleague for help."
            )
        else:
            answer = (
                "That verification did not match. Please provide either the "
                "last four digits of the registered phone number or the delivery "
                "pincode. You have one retry remaining."
            )
        return self._response(answer)

    def _verification_request(self) -> str:
        return (
            "I can help with that order. Please provide either the last four "
            "digits of the registered phone number or the six-digit delivery "
            "pincode for verification."
        )

    @staticmethod
    def _verification_failure() -> str:
        return (
            "I can't share order details because verification was unsuccessful. "
            "Please contact a support colleague for help."
        )

    @staticmethod
    def _should_escalate(message: str) -> bool:
        """Detect requests that require a human handover before generation."""

        text = message.casefold()
        escalation_phrases = (
            "speak to a person",
            "talk to a person",
            "human agent",
            "human support",
            "someone from support",
            "kisi insaan",
            "किसी इंसान",
            "मानव सहायता",
            "third time",
            "teesri baar",
            "तीसरी बार",
            "replacement approve",
            "approve kar",
            "approve kar do",
            "approve the replacement",
            "damage report",
            "damaged tv",
            "compensation",
            "compensate me",
            "pay me for",
            "monetary compensation",
            "muawza",
            "मुआवज़ा",
            "मुआवजा",
        )
        return any(phrase in text for phrase in escalation_phrases)

    @staticmethod
    def _escalation_reason(message: str) -> str:
        text = message.casefold()
        if any(
            phrase in text
            for phrase in ("third time", "teesri baar", "तीसरी बार")
        ):
            return "repeat_contact"
        if any(
            phrase in text
            for phrase in (
                "damage",
                "damaged",
                "replacement",
                "डैमेज",
                "refund",
                "compensation",
                "compensate",
                "muawza",
                "मुआव",
                "रिफ़ंड",
                "रिफंड",
            )
        ):
            return "refund_or_damage_review"
        return "customer_requested_human_support"

    def _verified_reference_number(self) -> str | None:
        """Return a verified refund reference without exposing other fields."""

        if not self.state.verified_order:
            return None
        for key in ("return_rrn", "cancellation_rrn", "rto_rrn"):
            raw_value = self.state.verified_order.get(key)
            if raw_value is not None:
                value = str(raw_value).strip()
                if value and value.casefold() != "none":
                    return value
        return None

    @staticmethod
    def _detect_topic(message: str) -> str:
        """Classify the latest message so topic changes replace stale context."""

        text = message.casefold()
        topic_terms = (
            ("delivery", ("delivery", "delivered", "late", "shipment", "डिलीवरी")),
            ("return", ("return", "exchange", "वापस", "रिटर्न", "एक्सचेंज")),
            ("refund", ("refund", "rrn", "रिफंड", "रिफ़ंड")),
            ("cancellation", ("cancel", "cancellation", "कैंसल", "रद्द")),
            ("damage", ("damage", "damaged", "broken", "डैमेज", "टूटा")),
            ("product", ("product", "sku", "price", "pincode", "उत्पाद")),
            (
                "verification",
                (
                    "verify",
                    "verification",
                    "phone suffix",
                    "last four",
                    "last 4",
                    "pincode",
                ),
            ),
        )
        for topic, terms in topic_terms:
            if any(term in text for term in terms):
                return topic
        return "general"

    def _record(self, role: str, content: str) -> None:
        self.state.conversation_history.append(
            {"role": role, "content": self._redact_secret(content)}
        )

    def _redact_secret(self, content: str) -> str:
        candidate = self._verification_secret
        if candidate is None and self.state.verification_status == "pending":
            candidate = self._extract_verification_value(
                content,
                self._extract_order_id(content),
            )
        if candidate:
            return re.sub(
                rf"(?<!\d){re.escape(candidate)}(?!\d)",
                "[redacted verification value]", content,
            )
        return content

    @staticmethod
    def _needs_order_id(message: str) -> bool:
        text = message.casefold()
        # General policy questions are answerable without customer identity.
        if re.search(r"\b(policy|policies|generally|standard|usually|window)\b|नीति|नियम", text):
            return False
        if re.search(r"\b(a|an|any)\s+order\b", text):
            return False
        if "refund" in text and re.search(
            r"\d{4}-\d{2}-\d{2}|\d{1,2}\s+\S+\s+\d{4}", text
        ):
            # A supplied initiation date and payment method describe a public
            # timeline calculation rather than a request to reveal an order.
            if re.search(r"upi|card|banking|wallet|neft|transfer", text):
                return False
        return bool(
            re.search(r"\b(my|our|mera|meri|mere|hamara)\b.*\b(order|parcel|delivery|refund|return|item)\b", text)
            or re.search(r"\b(order|parcel|refund)\b.*\b(not arrived|missing|late|status|cancel)\b", text)
            or re.search(r"(मेरा|मेरी|मेरे).*(ऑर्डर|पार्सल|डिलीवरी|रिफंड|रिफ़ंड)", text)
        )

    @staticmethod
    def _extract_order_id(message: str) -> str | None:
        for match in re.finditer(r"(?<!\d)(\d(?:[\s-]?\d){7})(?!\d)", message):
            value = re.sub(r"\D", "", match.group(1))
            if len(value) == 8:
                return value
        return None

    @staticmethod
    def _extract_verification_value(
        message: str,
        order_id: str | None,
    ) -> str | None:
        candidates = re.findall(r"(?<!\d)(\d{4}|\d{6})(?!\d)", message)
        order_digits = order_id or ""
        for candidate in candidates:
            if candidate != order_digits:
                return candidate
        return None

    @staticmethod
    def _detect_language(message: str, current: str) -> str:
        if re.search(r"[\u0900-\u097f]", message):
            return "hi"
        if re.search(r"\b(hai|mujhe|kaise|kya|chahiye|mera|meri|kab|nahi)\b", message.casefold()):
            return "hi-en"
        return current

    @staticmethod
    def _load_default_ticket_creator() -> TicketCreator:
        tools_path = (
            Path(__file__).resolve().parents[2]
            / "Capstone_Project_Dataset_FTTT_2026"
            / "tools"
            / "tools.py"
        )
        if not tools_path.is_file():
            raise FileNotFoundError(f"Offline support tools are missing: {tools_path}")
        spec = importlib.util.spec_from_file_location(
            "zentromart_offline_tools_for_conversation", tools_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load offline support tools: {tools_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.create_ticket

    def _response(
        self,
        answer: str,
        tool_response: ToolAwareResponse | None = None,
    ) -> ConversationResponse:
        if not self.state.introduced:
            introduction = {
                "hi": "मैं ZentroMart की AI सहायता सहायक हूँ। ",
                "hi-en": "Main ZentroMart ki AI support assistant hoon. ",
            }.get(self.state.customer_language, "I am ZentroMart's AI support assistant. ")
            if not re.search(r"\bAI\b|artificial intelligence", answer, re.IGNORECASE):
                answer = introduction + answer
            self.state.introduced = True
        answer = self._redact_secret(answer)
        calls = list(self._turn_calls)
        results = dict(self._turn_results)
        if tool_response is None:
            tool_response = ToolAwareResponse(
                answer=answer, sources=[], retrieval_scores=[], tool_results={},
                abstained=False,
            )
        else:
            # Reusing a verified record is not a new lookup invocation.
            calls.extend(call for call in tool_response.tool_calls if call != "lookup_order")
            results = {**tool_response.tool_results, **results}
        tool_response = replace(
            tool_response, answer=answer, tool_calls=list(dict.fromkeys(calls)),
            tool_results=results,
            ticket_id=self.state.ticket_id,
            needs_human=tool_response.needs_human or self.state.escalation_state == "escalated",
            state_changes={
                **tool_response.state_changes,
                "verification_status": self.state.verification_status,
                "escalation_state": self.state.escalation_state,
                "ticket_id": self.state.ticket_id,
            },
        )
        self._record("assistant", answer)
        return ConversationResponse(
            answer=answer,
            state=self.state.public_snapshot(),
            tool_response=tool_response,
        )


__all__ = [
    "ConversationResponse",
    "ConversationSession",
    "ConversationState",
    "TicketCreator",
]
