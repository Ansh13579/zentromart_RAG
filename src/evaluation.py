"""Evaluation for the Step 1 LLM-only support-agent baseline."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Any

if TYPE_CHECKING:
    if __package__:
        from .agent import SupportAgent
    else:
        from agent import SupportAgent


DEFAULT_QUESTIONS_PATH = (
    Path(__file__).resolve().parents[2]
    / "Capstone_Project_Dataset_FTTT_2026"
    / "practice"
    / "questions.csv"
)
DEFAULT_CONVERSATIONS_PATH = (
    Path(__file__).resolve().parents[2]
    / "Capstone_Project_Dataset_FTTT_2026"
    / "practice"
    / "conversations.csv"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "evaluations"


class Agent(Protocol):
    def respond(self, customer_question: str) -> str:
        """Generate a response for one customer question."""


class RAGAgent(Protocol):
    def respond_with_rag(
        self,
        customer_question: str,
        *,
        top_k: int = 5,
    ) -> Any:
        """Generate a grounded answer with retrieval trace."""


@dataclass
class ConversationTurnResult:
    conversation_id: str
    turn: int
    customer_message: str
    expected_decision: str
    predicted_decision: str
    decision_correct: bool
    response: str
    required_facts_found: list[str]
    required_facts_missing: list[str]
    forbidden_phrases_found: list[str]
    data_leakage_found: list[str]
    expected_tool_calls: list[str]
    actual_tool_calls: list[str]
    tool_calls_correct: bool
    verification_sequence_correct: bool
    passed: bool
    error: str | None = None


@dataclass
class ConversationResult:
    conversation_id: str
    scenario: str
    turns: list[dict[str, Any]]
    passed: bool
    failure_reasons: list[str]


@dataclass
class QuestionResult:
    question_id: str
    question: str
    expected_decision: str
    acceptable_decisions: list[str]
    predicted_decision: str
    decision_correct: bool
    response: str
    required_facts_found: list[str]
    required_facts_missing: list[str]
    forbidden_phrases_found: list[str]
    passed: bool
    error: str | None = None


@dataclass
class RAGQuestionResult:
    question_id: str
    question: str
    answerable: bool
    expected_decision: str
    acceptable_decisions: list[str]
    predicted_decision: str
    decision_correct: bool
    response: str
    source_section_ids: list[str]
    gold_evidence: list[str]
    evidence_sections_found: list[str]
    retrieval_hit: bool
    reciprocal_rank: float
    abstained: bool
    abstention_correct: bool
    required_facts_found: list[str]
    required_facts_missing: list[str]
    forbidden_phrases_found: list[str]
    grounded_answer_correct: bool
    passed: bool
    error: str | None = None


def _split_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(" ; ") if item.strip()]


def _gold_section_ids(value: str) -> list[str]:
    """Extract section IDs from gold evidence entries."""

    return [
        evidence
        for evidence in _split_values(value)
        if not evidence.lower().startswith("products.csv:")
    ]


def _normalise(value: str) -> str:
    normalized = (
        value.casefold()
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u02bc", "'")
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _contains_fact(response: str, fact: str) -> bool:
    """Check a required fact, accepting alternatives separated by '/'. """

    response_text = _normalise(response)
    alternatives = [item.strip() for item in fact.split(" / ") if item.strip()]
    return any(_normalise(item) in response_text for item in alternatives)


def _write_failure_analysis(
    output_dir: Path,
    stage: str,
    records: list[dict[str, Any]],
) -> None:
    """Persist the required input/expected/actual/root-cause/fix fields."""

    path = output_dir / "failure_analysis.json"
    if path.is_file():
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {}
    else:
        report = {}
    if not isinstance(report, dict):
        report = {}
    report[stage] = records
    report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_failure_analysis_from_saved_results(output_dir: Path) -> Path:
    """Create analysis for existing failure JSON without making model calls."""

    stages: dict[str, list[dict[str, Any]]] = {}
    files = (
        ("step1", "step1_failures.json"),
        ("step2", "step2_failures.json"),
    )
    for stage, filename in files:
        path = output_dir / filename
        if not path.is_file():
            continue
        try:
            failures = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(failures, list):
            continue
        records: list[dict[str, Any]] = []
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            rag = stage == "step2"
            categories: list[str] = []
            if failure.get("error"):
                categories.append("runtime_or_provider_error")
            if rag and failure.get("answerable") and not failure.get("retrieval_hit"):
                categories.append("retrieval_miss")
            if not failure.get("decision_correct", False):
                categories.append("decision_mismatch")
            missing = failure.get("required_facts_missing", [])
            if missing:
                categories.append("required_fact_omission")
            if failure.get("forbidden_phrases_found"):
                categories.append("forbidden_claim")
            if rag and not failure.get("answerable") and not failure.get("abstention_correct"):
                categories.append("unsupported_question_not_abstained")
            if rag and failure.get("answerable") and not failure.get("grounded_answer_correct"):
                categories.append("grounding_failure")
            root_cause = (
                "The answerable case's gold section was absent from the retrieved top-k."
                if "retrieval_miss" in categories
                else "The response failed one or more expected-decision, required-fact, abstention, or grounding checks."
            )
            planned_fix = (
                "Improve hybrid retrieval and query normalization; verify that all supporting sections enter the context."
                if "retrieval_miss" in categories
                else "Use evidence to cover every required fact and abstain only when the supplied evidence is insufficient."
            )
            records.append(
                {
                    "stage": stage,
                    "case_id": failure.get("question_id"),
                    "input": failure.get("question"),
                    "expected_behavior": {
                        "decision": failure.get("expected_decision"),
                        "acceptable_decisions": failure.get("acceptable_decisions", []),
                        "answerable": failure.get("answerable") if rag else None,
                        "required_facts": list(
                            dict.fromkeys(
                                failure.get("required_facts_found", []) + missing
                            )
                        ),
                        "gold_section_ids": failure.get("gold_evidence", []),
                    },
                    "actual_behavior": {
                        "decision": failure.get("predicted_decision"),
                        "response": failure.get("response"),
                        "retrieved_section_ids": failure.get("source_section_ids", []),
                        "required_facts_missing": missing,
                        "error": failure.get("error"),
                    },
                    "failure_category": categories or ["other"],
                    "root_cause": root_cause,
                    "planned_fix": planned_fix,
                }
            )
        stages[stage] = records

    conversation_path = output_dir / "conversation_failures.json"
    if conversation_path.is_file():
        try:
            failed_conversations = json.loads(
                conversation_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            failed_conversations = []
        expected_turns: dict[tuple[str, str], dict[str, str]] = {}
        if DEFAULT_CONVERSATIONS_PATH.is_file():
            with DEFAULT_CONVERSATIONS_PATH.open(
                "r", encoding="utf-8-sig", newline=""
            ) as file:
                for row in csv.DictReader(file):
                    expected_turns[(row["conversation_id"], row["turn"])] = row
        records = []
        for conversation in failed_conversations if isinstance(failed_conversations, list) else []:
            for turn in conversation.get("turns", []):
                if turn.get("passed"):
                    continue
                categories: list[str] = []
                if turn.get("error"):
                    categories.append("runtime_or_provider_error")
                if not turn.get("decision_correct"):
                    categories.append("decision_mismatch")
                if turn.get("required_facts_missing"):
                    categories.append("required_fact_omission")
                if turn.get("forbidden_phrases_found"):
                    categories.append("forbidden_claim")
                if turn.get("data_leakage_found"):
                    categories.append("privacy_leakage")
                if not turn.get("tool_calls_correct"):
                    categories.append("tool_call_mismatch")
                if not turn.get("verification_sequence_correct"):
                    categories.append("verification_sequence_error")
                records.append(
                    {
                        "stage": "conversations",
                        "case_id": f"{conversation.get('conversation_id')}:turn_{turn.get('turn')}",
                        "input": turn.get("customer_message"),
                        "expected_behavior": {
                            "decision": turn.get("expected_decision"),
                            "required_facts": _split_values(
                                expected_turns.get(
                                    (str(conversation.get("conversation_id")), str(turn.get("turn"))),
                                    {},
                                ).get("must_include", "")
                            ),
                            "forbidden_phrases": _split_values(
                                expected_turns.get(
                                    (str(conversation.get("conversation_id")), str(turn.get("turn"))),
                                    {},
                                ).get("must_not_include", "")
                            ),
                            "tool_calls": turn.get("expected_tool_calls", []),
                        },
                        "actual_behavior": {
                            "decision": turn.get("predicted_decision"),
                            "response": turn.get("response"),
                            "required_facts_missing": turn.get("required_facts_missing", []),
                            "tool_calls": turn.get("actual_tool_calls", []),
                            "error": turn.get("error"),
                        },
                        "failure_category": categories or ["other"],
                        "root_cause": "Observed conversation turn failed the listed evaluator checks; review the turn context and trace to confirm the underlying cause.",
                        "planned_fix": "Use the full conversation state, current policy evidence, and verified tool results to satisfy the failing checks.",
                    }
                )
        stages["conversations"] = records

    target = output_dir / "failure_analysis.json"
    stages["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    target.write_text(
        json.dumps(stages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def recheck_saved_step2_results(output_dir: Path) -> dict[str, Any]:
    """Re-score saved Step 2 responses after evaluator normalization changes.

    This path makes no LLM or embedding calls; it recalculates evaluator fields
    from the responses and retrieval traces already saved in step2_questions.json.
    """

    questions_path = output_dir / "step2_questions.json"
    metrics_path = output_dir / "step2_metrics.json"
    if not questions_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            "Saved Step 2 questions and metrics are required for offline re-scoring."
        )
    rows = json.loads(questions_path.read_text(encoding="utf-8"))
    old_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Saved Step 2 questions file must contain a list.")

    if __package__:
        from .agent import SupportAgent
    else:
        from agent import SupportAgent

    for row in rows:
        response = str(row.get("response", ""))
        row["abstained"] = SupportAgent._is_abstention(response)
        row["abstention_correct"] = (
            row["abstained"]
            if not row.get("answerable", False)
            else not row["abstained"]
        )
        known_facts = list(
            dict.fromkeys(
                row.get("required_facts_found", [])
                + row.get("required_facts_missing", [])
            )
        )
        row["required_facts_found"] = [
            fact for fact in known_facts if _contains_fact(response, fact)
        ]
        row["required_facts_missing"] = [
            fact for fact in known_facts if fact not in row["required_facts_found"]
        ]
        row["grounded_answer_correct"] = (
            bool(row.get("decision_correct"))
            and not row["required_facts_missing"]
            and not row.get("forbidden_phrases_found")
            and (
                bool(row.get("retrieval_hit"))
                if row.get("answerable")
                else bool(row["abstained"])
            )
        )
        row["passed"] = (
            row.get("error") is None
            and row["grounded_answer_correct"]
            and row["abstention_correct"]
        )

    total = len(rows)
    answerable = [row for row in rows if row.get("answerable")]
    unsupported = [row for row in rows if not row.get("answerable")]
    passed_count = sum(bool(row.get("passed")) for row in rows)
    old_metrics.update(
        {
            "rescored_from_saved_responses": True,
            "rescored_at_utc": datetime.now(timezone.utc).isoformat(),
            "passed_questions": passed_count,
            "failed_questions": total - passed_count,
            "pass_rate": passed_count / total if total else 0.0,
            "retrieval_hit_rate": (
                sum(bool(row.get("retrieval_hit")) for row in answerable)
                / len(answerable)
                if answerable
                else 0.0
            ),
            "mean_reciprocal_rank": (
                sum(float(row.get("reciprocal_rank", 0.0)) for row in answerable)
                / len(answerable)
                if answerable
                else 0.0
            ),
            "decision_accuracy": (
                sum(bool(row.get("decision_correct")) for row in rows) / total
                if total
                else 0.0
            ),
            "grounded_answer_accuracy": (
                sum(bool(row.get("grounded_answer_correct")) for row in rows) / total
                if total
                else 0.0
            ),
            "unsupported_abstention_accuracy": (
                sum(bool(row.get("abstention_correct")) for row in unsupported)
                / len(unsupported)
                if unsupported
                else 0.0
            ),
            "required_fact_accuracy": (
                sum(not row.get("required_facts_missing") for row in rows) / total
                if total
                else 0.0
            ),
            "forbidden_phrase_free_rate": (
                sum(not row.get("forbidden_phrases_found") for row in rows) / total
                if total
                else 0.0
            ),
        }
    )

    questions_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(old_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    failures = [row for row in rows if not row.get("passed")]
    (output_dir / "step2_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "retrieval_metrics.json").write_text(
        json.dumps(
            {
                "top_k": old_metrics.get("top_k", 5),
                "answerable_questions": len(answerable),
                "retrieval_hit_rate": old_metrics["retrieval_hit_rate"],
                "mean_reciprocal_rank": old_metrics["mean_reciprocal_rank"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    step1_path = output_dir / "step1_metrics.json"
    if step1_path.is_file():
        save_step_comparison(
            json.loads(step1_path.read_text(encoding="utf-8")),
            old_metrics,
            output_dir / "step1_vs_step2.json",
        )
    build_failure_analysis_from_saved_results(output_dir)
    return old_metrics


def _question_failure_analysis(
    result: QuestionResult | RAGQuestionResult,
    *,
    stage: str,
) -> dict[str, Any]:
    """Translate a failed question result into an actionable failure record."""

    categories: list[str] = []
    if result.error:
        categories.append("runtime_or_provider_error")
    if not result.decision_correct:
        categories.append("decision_mismatch")
    if result.required_facts_missing:
        categories.append("required_fact_omission")
    if result.forbidden_phrases_found:
        categories.append("forbidden_claim")

    if isinstance(result, RAGQuestionResult):
        if result.answerable and not result.retrieval_hit:
            categories.append("retrieval_miss")
        if not result.answerable and not result.abstention_correct:
            categories.append("unsupported_question_not_abstained")
        if result.answerable and not result.grounded_answer_correct:
            categories.append("grounding_failure")
        root_cause = (
            "The answerable case's gold section was absent from the retrieved top-k."
            if "retrieval_miss" in categories
            else "The answer did not satisfy at least one expected decision, required-fact, or grounding check."
        )
        planned_fix = (
            "Improve hybrid retrieval and query normalization, then ensure all required evidence sections are included."
            if "retrieval_miss" in categories
            else "Use the retrieved evidence to answer every required sub-question and abstain only when the evidence is insufficient."
        )
        actual: dict[str, Any] = {
            "decision": result.predicted_decision,
            "response": result.response,
            "retrieved_section_ids": result.source_section_ids,
            "gold_section_ids": result.gold_evidence,
            "required_facts_missing": result.required_facts_missing,
        }
        expected: dict[str, Any] = {
            "decision": result.expected_decision,
            "acceptable_decisions": result.acceptable_decisions,
            "answerable": result.answerable,
            "required_facts": result.required_facts_missing + result.required_facts_found,
            "gold_section_ids": result.gold_evidence,
        }
    else:
        categories.append("llm_only_baseline_gap")
        root_cause = (
            "Step 1 intentionally answers without policy retrieval or structured data; this is the measured LLM-only baseline."
        )
        planned_fix = (
            "Use Step 2 evidence-grounded retrieval for policy answers; keep this failure as a baseline comparison."
        )
        actual = {
            "decision": result.predicted_decision,
            "response": result.response,
            "required_facts_missing": result.required_facts_missing,
        }
        expected = {
            "decision": result.expected_decision,
            "acceptable_decisions": result.acceptable_decisions,
            "required_facts": result.required_facts_missing + result.required_facts_found,
        }

    return {
        "stage": stage,
        "case_id": result.question_id,
        "input": result.question,
        "expected_behavior": expected,
        "actual_behavior": actual,
        "failure_category": categories,
        "root_cause": root_cause,
        "planned_fix": planned_fix,
        "error": result.error,
    }


def _predict_decision(response: str) -> str:
    """Classify the response using transparent baseline heuristics."""

    text = _normalise(response)
    refusal_terms = (
        "i don't know",
        "i do not know",
        "cannot confirm",
        "don't have information",
        "do not have information",
        "pata nahi",
        "jaankari nahi",
        "जानकारी नहीं",
    )
    escalation_terms = (
        "human colleague",
        "connect you to a colleague",
        "escalate",
        "human support",
        "colleague",
        "handed this over",
        "handed over",
        "support team for review",
        "ticket",
        "सहयोगी",
        "connect",
    )
    clarification_terms = (
        "could you provide",
        "please provide",
        "please share",
        "share your",
        "may i have",
        "what is your",
        "which order",
        "order id",
        "eight-digit order",
        "कृपया बताएं",
    )

    if any(term in text for term in refusal_terms):
        return "refuse"
    if any(term in text for term in escalation_terms):
        return "escalate"
    if any(term in text for term in clarification_terms):
        return "clarify"
    return "answer"


def evaluate_questions(
    questions_path: Path,
    output_dir: Path,
    agent: Agent | None = None,
) -> dict[str, object]:
    """Run the agent over the practice questions and save evaluation files."""

    if not questions_path.is_file():
        raise FileNotFoundError(
            f"Practice questions file was not found: {questions_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if agent is None:
        if __package__:
            from .agent import SupportAgent
        else:
            from agent import SupportAgent

        evaluator_agent: Agent = SupportAgent()
    else:
        evaluator_agent = agent
    results: list[QuestionResult] = []

    with questions_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        question_id = row["id"]
        response = ""
        error: str | None = None

        try:
            response = evaluator_agent.respond(row["question"])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        acceptable_decisions = _split_values(row["acceptable_decisions"])
        predicted_decision = _predict_decision(response) if response else "error"
        decision_correct = predicted_decision in acceptable_decisions

        required_facts = _split_values(row["must_include"])
        required_facts_found = [
            fact for fact in required_facts if _contains_fact(response, fact)
        ]
        required_facts_missing = [
            fact for fact in required_facts if fact not in required_facts_found
        ]

        forbidden_phrases = _split_values(row["must_not_include"])
        forbidden_phrases_found = [
            phrase
            for phrase in forbidden_phrases
            if _normalise(phrase) in _normalise(response)
        ]

        passed = (
            error is None
            and decision_correct
            and not required_facts_missing
            and not forbidden_phrases_found
        )
        results.append(
            QuestionResult(
                question_id=question_id,
                question=row["question"],
                expected_decision=row["expected_decision"],
                acceptable_decisions=acceptable_decisions,
                predicted_decision=predicted_decision,
                decision_correct=decision_correct,
                response=response,
                required_facts_found=required_facts_found,
                required_facts_missing=required_facts_missing,
                forbidden_phrases_found=forbidden_phrases_found,
                passed=passed,
                error=error,
            )
        )

    passed_count = sum(result.passed for result in results)
    decision_correct_count = sum(result.decision_correct for result in results)
    results_as_dicts = [asdict(result) for result in results]
    metrics = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "questions_path": str(questions_path),
        "total_questions": len(results),
        "passed_questions": passed_count,
        "failed_questions": len(results) - passed_count,
        "pass_rate": passed_count / len(results) if results else 0.0,
        "decision_accuracy": (
            decision_correct_count / len(results) if results else 0.0
        ),
        "required_fact_accuracy": (
            sum(not result.required_facts_missing for result in results)
            / len(results)
            if results
            else 0.0
        ),
        "forbidden_phrase_free_rate": (
            sum(not result.forbidden_phrases_found for result in results)
            / len(results)
            if results
            else 0.0
        ),
    }
    failures = [result for result in results_as_dicts if not result["passed"]]

    (output_dir / "step1_questions.json").write_text(
        json.dumps(results_as_dicts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "step1_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "step1_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_failure_analysis(
        output_dir,
        "step1",
        [
            _question_failure_analysis(result, stage="step1")
            for result in results
            if not result.passed
        ],
    )
    return metrics


def evaluate_rag_questions(
    questions_path: Path,
    output_dir: Path,
    agent: RAGAgent | None = None,
    *,
    top_k: int = 5,
) -> dict[str, object]:
    """Evaluate retrieval and grounded answers for all practice questions."""

    if not questions_path.is_file():
        raise FileNotFoundError(
            f"Practice questions file was not found: {questions_path}"
        )
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if agent is None:
        if __package__:
            from .agent import SupportAgent
        else:
            from agent import SupportAgent

        rag_agent: RAGAgent = SupportAgent()
    else:
        rag_agent = agent

    with questions_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    results: list[RAGQuestionResult] = []
    for row in rows:
        response = ""
        error: str | None = None
        source_section_ids: list[str] = []
        retrieval_scores: list[float] = []
        abstained = False

        try:
            rag_response = rag_agent.respond_with_rag(
                row["question"],
                top_k=top_k,
            )
            response = rag_response.answer
            source_section_ids = [
                str(source.get("section_id", ""))
                for source in (
                    getattr(rag_response, "retrieved_sources", None)
                    or rag_response.sources
                )
                if source.get("section_id")
            ]
            retrieval_scores = list(rag_response.retrieval_scores)
            abstained = bool(rag_response.abstained)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        acceptable_decisions = _split_values(row["acceptable_decisions"])
        predicted_decision = _predict_decision(response) if response else "error"
        decision_correct = predicted_decision in acceptable_decisions
        gold_evidence = _gold_section_ids(row["gold_evidence"])
        found_sections = [
            section for section in source_section_ids if section in gold_evidence
        ]
        first_matching_rank = next(
            (
                rank
                for rank, section in enumerate(source_section_ids, start=1)
                if section in gold_evidence
            ),
            None,
        )
        retrieval_hit = bool(found_sections)
        reciprocal_rank = 1 / first_matching_rank if first_matching_rank else 0.0

        required_facts = _split_values(row["must_include"])
        required_facts_found = [
            fact for fact in required_facts if _contains_fact(response, fact)
        ]
        required_facts_missing = [
            fact for fact in required_facts if fact not in required_facts_found
        ]
        forbidden_phrases = _split_values(row["must_not_include"])
        forbidden_phrases_found = [
            phrase
            for phrase in forbidden_phrases
            if _normalise(phrase) in _normalise(response)
        ]
        answerable = row["answerable"].strip().casefold() == "true"
        abstention_correct = abstained if not answerable else not abstained
        grounded_answer_correct = (
            decision_correct
            and not required_facts_missing
            and not forbidden_phrases_found
            and (retrieval_hit if answerable else abstained)
        )
        passed = error is None and grounded_answer_correct and abstention_correct

        results.append(
            RAGQuestionResult(
                question_id=row["id"],
                question=row["question"],
                answerable=answerable,
                expected_decision=row["expected_decision"],
                acceptable_decisions=acceptable_decisions,
                predicted_decision=predicted_decision,
                decision_correct=decision_correct,
                response=response,
                source_section_ids=source_section_ids,
                gold_evidence=gold_evidence,
                evidence_sections_found=found_sections,
                retrieval_hit=retrieval_hit,
                reciprocal_rank=reciprocal_rank,
                abstained=abstained,
                abstention_correct=abstention_correct,
                required_facts_found=required_facts_found,
                required_facts_missing=required_facts_missing,
                forbidden_phrases_found=forbidden_phrases_found,
                grounded_answer_correct=grounded_answer_correct,
                passed=passed,
                error=error,
            )
        )

    result_dicts = [asdict(result) for result in results]
    answerable_results = [result for result in results if result.answerable]
    failures = [result for result in result_dicts if not result["passed"]]
    metrics = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "questions_path": str(questions_path),
        "top_k": top_k,
        "total_questions": len(results),
        "passed_questions": sum(result.passed for result in results),
        "failed_questions": len(results) - sum(result.passed for result in results),
        "pass_rate": (
            sum(result.passed for result in results) / len(results)
            if results
            else 0.0
        ),
        "retrieval_hit_rate": (
            sum(result.retrieval_hit for result in answerable_results)
            / len(answerable_results)
            if answerable_results
            else 0.0
        ),
        "mean_reciprocal_rank": (
            sum(result.reciprocal_rank for result in answerable_results)
            / len(answerable_results)
            if answerable_results
            else 0.0
        ),
        "decision_accuracy": (
            sum(result.decision_correct for result in results) / len(results)
            if results
            else 0.0
        ),
        "grounded_answer_accuracy": (
            sum(result.grounded_answer_correct for result in results)
            / len(results)
            if results
            else 0.0
        ),
        "unsupported_abstention_accuracy": (
            sum(
                result.abstention_correct
                for result in results
                if not result.answerable
            )
            / sum(not result.answerable for result in results)
            if any(not result.answerable for result in results)
            else 0.0
        ),
        "required_fact_accuracy": (
            sum(not result.required_facts_missing for result in results)
            / len(results)
            if results
            else 0.0
        ),
        "forbidden_phrase_free_rate": (
            sum(not result.forbidden_phrases_found for result in results)
            / len(results)
            if results
            else 0.0
        ),
    }
    (output_dir / "step2_questions.json").write_text(
        json.dumps(result_dicts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "step2_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "retrieval_metrics.json").write_text(
        json.dumps(
            {
                "top_k": top_k,
                "answerable_questions": len(answerable_results),
                "retrieval_hit_rate": metrics["retrieval_hit_rate"],
                "mean_reciprocal_rank": metrics["mean_reciprocal_rank"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "step2_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_failure_analysis(
        output_dir,
        "step2",
        [
            _question_failure_analysis(result, stage="step2")
            for result in results
            if not result.passed
        ],
    )
    return metrics


def _expected_tool_names(value: str) -> list[str]:
    """Extract ordered tool names from the practice expectation."""

    return re.findall(r"(lookup_order|create_ticket)\s*\(", value)


def _normalise_tool_call(name: str, order_id: str | None = None) -> str:
    if name == "lookup_order" and order_id:
        return f"lookup_order(order_id={order_id})"
    return f"{name}()"


def _tool_call_names(calls: list[str]) -> list[str]:
    return [
        "lookup_order" if call.startswith("lookup_order") else "create_ticket"
        for call in calls
    ]


def _verification_sequence_is_correct(
    actual_call_names: list[str],
    expected_call_names: list[str],
    verification_attempts: int,
) -> bool:
    """Check that order lookup precedes details and retries stay bounded."""

    if "lookup_order" in expected_call_names and "lookup_order" not in actual_call_names:
        return False
    if "create_ticket" in actual_call_names and "lookup_order" in expected_call_names:
        return verification_attempts <= 2
    return verification_attempts <= 2


def _conversation_data_leaks(
    response: str,
    *,
    verified: bool,
    order_result: dict[str, Any] | None,
    verification_values: list[str],
) -> list[str]:
    """Detect private order fields appearing before verification."""

    if verified or not order_result:
        return []
    response_text = _normalise(response)
    leaks: list[str] = []
    for key in (
        "status",
        "total_inr",
        "delivery_pincode",
        "customer_first_name",
        "subtotal_inr",
        "payment_method",
    ):
        value = str(order_result.get(key, "")).strip()
        if value and _normalise(value) in response_text:
            leaks.append(key)
    for value in verification_values:
        if value and value in re.sub(r"\D", "", response):
            leaks.append("verification_secret")
            break
    return leaks


def evaluate_conversations(
    conversations_path: Path,
    output_dir: Path,
    *,
    agent: Any | None = None,
) -> dict[str, object]:
    """Evaluate multi-turn behavior against practice/conversations.csv."""

    if not conversations_path.is_file():
        raise FileNotFoundError(
            f"Practice conversations file was not found: {conversations_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if agent is None:
        if __package__:
            from .agent import SupportAgent
            from .conversation import ConversationSession
        else:
            from agent import SupportAgent
            from conversation import ConversationSession

        base_agent = SupportAgent()
        session_factory = lambda lookup, ticket: ConversationSession(
            agent=base_agent,
            order_lookup=lookup,
            ticket_creator=ticket,
        )
    else:
        if __package__:
            from .conversation import ConversationSession
        else:
            from conversation import ConversationSession
        session_factory = lambda lookup, ticket: ConversationSession(
            agent=agent,
            order_lookup=lookup,
            ticket_creator=ticket,
        )

    with conversations_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["conversation_id"], []).append(row)

    conversation_results: list[ConversationResult] = []
    conversation_failure_analysis: list[dict[str, Any]] = []
    tool_call_traces: list[dict[str, Any]] = []
    total_conversations = len(grouped)
    for conversation_number, (conversation_id, conversation_rows) in enumerate(
        grouped.items(),
        start=1,
    ):
        print(
            f"Evaluating conversation {conversation_number}/{total_conversations}: "
            f"{conversation_id}",
            flush=True,
        )
        conversation_rows.sort(key=lambda row: int(row["turn"]))
        lookup_calls: list[str] = []
        ticket_calls: list[str] = []
        lookup_results: list[dict[str, Any]] = []
        verification_values: list[str] = []

        def lookup(order_id: str, verification_value: str | None = None) -> dict[str, Any]:
            call = _normalise_tool_call("lookup_order", order_id)
            if verification_value is not None:
                call = call[:-1] + ", verification_value=[redacted])"
            lookup_calls.append(call)
            if verification_value is not None:
                verification_values.append(
                    re.sub(r"\D", "", str(verification_value))
                )
            if __package__:
                from .agent import SupportAgent
            else:
                from agent import SupportAgent
            result = SupportAgent._load_default_order_lookup()(
                order_id,
                verification_value=verification_value,
            )
            lookup_results.append(result)
            return result

        def ticket(
            summary: str,
            order_id: str | None = None,
            reason: str | None = None,
            intent: str | None = None,
        ) -> dict[str, Any]:
            ticket_calls.append("create_ticket()")
            return {"created": True, "ticket_id": f"EVAL-{conversation_id}"}

        session = session_factory(lookup, ticket)
        turn_results: list[ConversationTurnResult] = []
        for row in conversation_rows:
            response = ""
            error: str | None = None
            expected_tools = _expected_tool_names(row["expected_tool_calls"])
            before_lookup_calls = len(lookup_calls)
            before_ticket_calls = len(ticket_calls)
            try:
                turn_response = session.process_message(row["customer_message"])
                response = turn_response.answer
                predicted_decision = _predict_decision(response)
            except Exception as exc:
                predicted_decision = "error"
                error = f"{type(exc).__name__}: {exc}"

            actual_calls = (
                lookup_calls[before_lookup_calls:]
                + ticket_calls[before_ticket_calls:]
            )
            expected_call_names = expected_tools
            actual_call_names = _tool_call_names(actual_calls)
            decision_correct = predicted_decision == row["expected_decision"]
            required = _split_values(row["must_include"])
            found = [fact for fact in required if _contains_fact(response, fact)]
            missing = [fact for fact in required if fact not in found]
            forbidden = _split_values(row["must_not_include"])
            forbidden_found = [
                phrase for phrase in forbidden if _normalise(phrase) in _normalise(response)
            ]
            verified = bool(
                getattr(session.state, "verification_status", "") == "verified"
            )
            latest_order = lookup_results[-1] if lookup_results else None
            leaks = _conversation_data_leaks(
                response,
                verified=verified,
                order_result=latest_order,
                verification_values=verification_values,
            )
            tool_correct = actual_call_names == expected_call_names
            verification_correct = _verification_sequence_is_correct(
                actual_call_names,
                expected_tools,
                session.state.verification_attempts,
            )
            passed = (
                error is None
                and decision_correct
                and not missing
                and not forbidden_found
                and not leaks
                and tool_correct
                and verification_correct
            )
            if not passed:
                categories: list[str] = []
                causes: list[str] = []
                fixes: list[str] = []
                if error:
                    categories.append("runtime_or_provider_error")
                    causes.append("The turn raised an exception before a valid response was produced.")
                    fixes.append("Resolve the recorded exception and rerun this case.")
                if not decision_correct:
                    categories.append("decision_mismatch")
                    causes.append("The response intent did not match the expected answer, clarification, refusal, or escalation.")
                    fixes.append("Make response intent explicit and review the relevant policy or verification branch.")
                if missing:
                    categories.append("required_fact_omission")
                    causes.append("The response omitted one or more rubric-required facts.")
                    fixes.append("Use retrieved policy and verified tool facts to state each required date, amount, condition, or next step.")
                if forbidden_found:
                    categories.append("forbidden_claim")
                    causes.append("The response included a phrase prohibited by the practice case.")
                    fixes.append("Remove the unsupported claim and constrain the response to evidence and verified tool results.")
                if leaks:
                    categories.append("privacy_leakage")
                    causes.append("A private order field or verification secret appeared before verification passed.")
                    fixes.append("Tighten redaction and keep private fields out of pre-verification prompts and traces.")
                if not tool_correct:
                    categories.append("tool_call_mismatch")
                    causes.append("The observed tool sequence differed from the case expectation.")
                    fixes.append("Review intent routing and invoke only the required trusted tool at the expected point.")
                if not verification_correct:
                    categories.append("verification_sequence_error")
                    causes.append("The verification retry count or lookup ordering was incorrect.")
                    fixes.append("Enforce the one-retry limit and require successful verification before disclosing details.")

                conversation_failure_analysis.append(
                    {
                        "stage": "conversations",
                        "case_id": f"{conversation_id}:turn_{row['turn']}",
                        "input": row["customer_message"],
                        "expected_behavior": {
                            "decision": row["expected_decision"],
                            "required_facts": required,
                            "forbidden_phrases": forbidden,
                            "tool_calls": expected_call_names,
                        },
                        "actual_behavior": {
                            "decision": predicted_decision,
                            "response": response,
                            "required_facts_missing": missing,
                            "forbidden_phrases_found": forbidden_found,
                            "data_leakage_fields": leaks,
                            "tool_calls": actual_call_names,
                            "error": error,
                        },
                        "failure_category": categories,
                        "root_cause": " ".join(causes),
                        "planned_fix": " ".join(fixes),
                    }
                )
            turn_results.append(
                ConversationTurnResult(
                    conversation_id=conversation_id,
                    turn=int(row["turn"]),
                    customer_message=row["customer_message"],
                    expected_decision=row["expected_decision"],
                    predicted_decision=predicted_decision,
                    decision_correct=decision_correct,
                    response=response,
                    required_facts_found=found,
                    required_facts_missing=missing,
                    forbidden_phrases_found=forbidden_found,
                    data_leakage_found=leaks,
                    expected_tool_calls=expected_call_names,
                    actual_tool_calls=actual_call_names,
                    tool_calls_correct=tool_correct,
                    verification_sequence_correct=verification_correct,
                    passed=passed,
                    error=error,
                )
            )
            tool_call_traces.append(
                {
                    "conversation_id": conversation_id,
                    "turn": int(row["turn"]),
                    "expected_tool_calls": expected_call_names,
                    "actual_tool_calls": actual_call_names,
                    "verification_status": session.state.verification_status,
                    "verification_attempts": session.state.verification_attempts,
                }
            )

        failures = [
            f"turn {result.turn}: {field}"
            for result in turn_results
            for field, failed in (
                ("decision", not result.decision_correct),
                ("required_facts", bool(result.required_facts_missing)),
                ("forbidden_phrase", bool(result.forbidden_phrases_found)),
                ("data_leakage", bool(result.data_leakage_found)),
                ("tool_calls", not result.tool_calls_correct),
                ("verification_sequence", not result.verification_sequence_correct),
                ("error", result.error is not None),
            )
            if failed
        ]
        conversation_result = ConversationResult(
            conversation_id=conversation_id,
            scenario=conversation_rows[0]["scenario"],
            turns=[asdict(result) for result in turn_results],
            passed=not failures,
            failure_reasons=failures,
        )
        conversation_results.append(conversation_result)
        print(
            f"Completed {conversation_id}: "
            f"{'passed' if conversation_result.passed else 'failed'}",
            flush=True,
        )

    all_turns = [
        turn
        for conversation in conversation_results
        for turn in conversation.turns
    ]
    passed_turns = sum(turn["passed"] for turn in all_turns)
    metrics = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conversations_path": str(conversations_path),
        "total_conversations": len(conversation_results),
        "passed_conversations": sum(result.passed for result in conversation_results),
        "failed_conversations": sum(not result.passed for result in conversation_results),
        "conversation_pass_rate": (
            sum(result.passed for result in conversation_results)
            / len(conversation_results)
            if conversation_results
            else 0.0
        ),
        "total_turns": len(all_turns),
        "passed_turns": passed_turns,
        "turn_pass_rate": passed_turns / len(all_turns) if all_turns else 0.0,
        "tool_call_accuracy": (
            sum(turn["tool_calls_correct"] for turn in all_turns) / len(all_turns)
            if all_turns
            else 0.0
        ),
        "verification_sequence_accuracy": (
            sum(turn["verification_sequence_correct"] for turn in all_turns)
            / len(all_turns)
            if all_turns
            else 0.0
        ),
        "follow_up_behavior_accuracy": (
            sum(
                turn["predicted_decision"] == "clarify"
                for turn in all_turns
                if turn["expected_decision"] == "clarify"
            )
            / sum(turn["expected_decision"] == "clarify" for turn in all_turns)
            if any(turn["expected_decision"] == "clarify" for turn in all_turns)
            else 0.0
        ),
        "clarification_accuracy": (
            sum(
                turn["decision_correct"]
                for turn in all_turns
                if turn["expected_decision"] == "clarify"
            )
            / sum(turn["expected_decision"] == "clarify" for turn in all_turns)
            if any(turn["expected_decision"] == "clarify" for turn in all_turns)
            else 0.0
        ),
        "escalation_accuracy": (
            sum(
                turn["decision_correct"]
                for turn in all_turns
                if turn["expected_decision"] == "escalate"
            )
            / sum(turn["expected_decision"] == "escalate" for turn in all_turns)
            if any(turn["expected_decision"] == "escalate" for turn in all_turns)
            else 0.0
        ),
        "ticket_creation_accuracy": (
            sum(
                "create_ticket" in turn["actual_tool_calls"]
                for turn in all_turns
                if "create_ticket" in turn["expected_tool_calls"]
            )
            / sum(
                "create_ticket" in turn["expected_tool_calls"] for turn in all_turns
            )
            if any("create_ticket" in turn["expected_tool_calls"] for turn in all_turns)
            else 0.0
        ),
        "data_leakage_free_rate": (
            sum(not turn["data_leakage_found"] for turn in all_turns) / len(all_turns)
            if all_turns
            else 0.0
        ),
    }
    conversation_dicts = [asdict(result) for result in conversation_results]
    failures = [result for result in conversation_dicts if not result["passed"]]
    (output_dir / "conversations.json").write_text(
        json.dumps(conversation_dicts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "conversation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "tool_call_traces.json").write_text(
        json.dumps(tool_call_traces, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "conversation_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_failure_analysis(
        output_dir,
        "conversations",
        conversation_failure_analysis,
    )
    return metrics


def save_step_comparison(
    step1_metrics: dict[str, object],
    step2_metrics: dict[str, object],
    output_path: Path,
) -> None:
    """Save a compact Step 1 versus Step 2 metric comparison."""

    comparison = {
        "step1": {
            "pass_rate": step1_metrics.get("pass_rate", 0.0),
            "decision_accuracy": step1_metrics.get("decision_accuracy", 0.0),
            "required_fact_accuracy": step1_metrics.get(
                "required_fact_accuracy", 0.0
            ),
            "forbidden_phrase_free_rate": step1_metrics.get(
                "forbidden_phrase_free_rate", 0.0
            ),
        },
        "step2": {
            "pass_rate": step2_metrics.get("pass_rate", 0.0),
            "decision_accuracy": step2_metrics.get("decision_accuracy", 0.0),
            "grounded_answer_accuracy": step2_metrics.get(
                "grounded_answer_accuracy", 0.0
            ),
            "retrieval_hit_rate": step2_metrics.get("retrieval_hit_rate", 0.0),
            "mean_reciprocal_rank": step2_metrics.get(
                "mean_reciprocal_rank", 0.0
            ),
            "required_fact_accuracy": step2_metrics.get(
                "required_fact_accuracy", 0.0
            ),
            "forbidden_phrase_free_rate": step2_metrics.get(
                "forbidden_phrase_free_rate", 0.0
            ),
        },
    }
    output_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Step 1, Step 2 and conversation agents."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="Path to practice/questions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for evaluation JSON files",
    )
    parser.add_argument(
        "--conversations",
        type=Path,
        default=DEFAULT_CONVERSATIONS_PATH,
        help="Path to practice/conversations.csv",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--step1-only",
        action="store_true",
        help="Run only the LLM-only baseline evaluation.",
    )
    parser.add_argument(
        "--step2-only",
        action="store_true",
        help="Run only the RAG evaluation.",
    )
    parser.add_argument(
        "--conversations-only",
        action="store_true",
        help="Run only the multi-turn conversation evaluation.",
    )
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Build failure_analysis.json from saved results without model calls.",
    )
    parser.add_argument(
        "--recheck-saved-step2",
        action="store_true",
        help="Recalculate Step 2 metrics from saved responses without model calls.",
    )
    args = parser.parse_args()

    if args.recheck_saved_step2:
        print(
            json.dumps(
                recheck_saved_step2_results(args.output_dir),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.failures_only:
        print(build_failure_analysis_from_saved_results(args.output_dir))
        return

    selected_modes = sum(
        [args.step1_only, args.step2_only, args.conversations_only]
    )
    if selected_modes > 1:
        parser.error("Only one --*-only option can be used at a time.")

    if args.step2_only:
        metrics = evaluate_rag_questions(
            args.questions,
            args.output_dir,
            top_k=args.top_k,
        )
        baseline_path = args.output_dir / "step1_metrics.json"
        if baseline_path.is_file():
            save_step_comparison(
                json.loads(baseline_path.read_text(encoding="utf-8")),
                metrics,
                args.output_dir / "step1_vs_step2.json",
            )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return
    if args.conversations_only:
        metrics = evaluate_conversations(args.conversations, args.output_dir)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    step1_metrics = evaluate_questions(args.questions, args.output_dir)
    if args.step1_only:
        print(json.dumps(step1_metrics, ensure_ascii=False, indent=2))
        return

    step2_metrics = evaluate_rag_questions(
        args.questions,
        args.output_dir,
        top_k=args.top_k,
    )
    comparison_path = args.output_dir / "step1_vs_step2.json"
    save_step_comparison(step1_metrics, step2_metrics, comparison_path)
    conversation_metrics = evaluate_conversations(
        args.conversations,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "step1": step1_metrics,
                "step2": step2_metrics,
                "conversations": conversation_metrics,
                "comparison_path": str(comparison_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
