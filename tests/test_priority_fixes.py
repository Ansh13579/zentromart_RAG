"""Regression checks for customer-visible flows; all model/speech calls are local fakes."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import soundfile as sf

from src.agent import SupportAgent, ToolAwareResponse
from src.conversation import ConversationSession
from src.data_store import DataStore
from src.policy_facts import add_calendar_evidence, calculate_policy_facts
from src.voice import VoiceService


def response(text="The verified order is on its way."):
    return ToolAwareResponse(
        answer=text, sources=[], retrieval_scores=[], tool_results={},
        abstained=False, tool_calls=["lookup_order"],
    )


def lookup(order_id, verification_value=None):
    if verification_value is None:
        return {"verification_required": True, "order_id": order_id}
    if verification_value != "4321":
        return {"verified": False, "verification_required": True}
    return {
        "found": True, "verified": True, "order_id": order_id,
        "status": "shipped", "items": [],
    }


class ConversationRegressionTests(unittest.TestCase):
    def setUp(self):
        self.agent = Mock()
        self.agent.respond_with_tools.return_value = response()
        self.lookup = Mock(side_effect=lookup)
        self.ticket = Mock(return_value={"created": True, "ticket_id": "MOCK-TICKET"})
        self.session = ConversationSession(
            agent=self.agent, order_lookup=self.lookup, ticket_creator=self.ticket,
        )

    def verify(self, order_id="12345678"):
        first = self.session.process_message(f"My order {order_id} is late.")
        self.assertIn("AI", first.answer)
        self.assertEqual(first.tool_response.tool_calls, ["lookup_order"])
        return self.session.process_message("The last four digits are 4321.")

    def test_verified_followup_does_not_repeat_lookup_or_ask_for_id(self):
        self.verify()
        calls = self.lookup.call_count
        result = self.session.process_message("Where is my order?")
        self.assertNotIn("share your", result.answer.casefold())
        self.assertEqual(self.lookup.call_count, calls)
        self.assertEqual(result.tool_response.tool_calls, [])
        self.assertEqual(
            self.agent.respond_with_tools.call_args.kwargs["verified_order"]["order_id"],
            "12345678",
        )

    def test_general_delivery_policy_does_not_request_an_order(self):
        result = self.session.process_message("What is your standard delivery policy?")
        self.lookup.assert_not_called()
        self.agent.respond_with_tools.assert_called_once()
        self.assertIn("AI", result.answer)

    def test_supplied_refund_date_and_method_need_no_private_lookup(self):
        self.session.process_message("My card refund started on 17 October 2026. By when?")
        self.lookup.assert_not_called()
        self.agent.respond_with_tools.assert_called_once()

    def test_switching_orders_drops_private_history_and_reverifies(self):
        self.verify()
        self.session.state.conversation_history.append(
            {"role": "assistant", "content": "Previous order PRIVATE-DETAIL"}
        )
        result = self.session.process_message("Please check order 87654321 instead.")
        self.assertEqual(result.state["verification_status"], "pending")
        self.assertIsNone(self.session.state.verified_order)
        self.assertNotIn("PRIVATE-DETAIL", json.dumps(result.state))
        self.assertNotIn("12345678", json.dumps(result.state))
        self.assertEqual(self.lookup.call_args.args[0], "87654321")
        self.assertNotIn("AI support assistant", result.answer)

    def test_failed_verification_cannot_reset_by_switching_away_and_back(self):
        self.session.process_message("Order 12345678")
        self.session.process_message("1111")
        self.session.process_message("2222")
        self.session.process_message("Order 87654321")
        result = self.session.process_message("Order 12345678")
        self.assertEqual(result.state["verification_status"], "failed")
        self.assertEqual(result.state["verification_attempts"], 2)

    def test_handover_preserves_ticket_call_and_context_without_secrets(self):
        self.verify()
        result = self.session.process_message("I want compensation for this delay.")
        self.assertEqual(result.tool_response.ticket_id, "MOCK-TICKET")
        self.assertEqual(result.tool_response.tool_calls, ["create_ticket"])
        self.assertTrue(result.tool_response.needs_human)
        ticket_context = self.ticket.call_args.kwargs["summary"]
        self.assertIn("12345678", ticket_context)
        self.assertIn("Reason:", ticket_context)
        self.assertNotIn("4321", ticket_context)
        self.assertTrue(result.tool_response.tool_results["create_ticket"]["created"])

    def test_failed_ticket_is_never_announced_as_created(self):
        self.ticket.return_value = {"created": False, "error": "disk unavailable"}
        with self.assertRaisesRegex(RuntimeError, "Ticket creation failed"):
            self.session.process_message("I want a human agent.")
        self.assertIsNone(self.session.state.ticket_id)
        self.assertNotIn("I created ticket", json.dumps(self.session.state.public_snapshot()))

    def test_refund_status_does_not_escalate_on_words_alone(self):
        self.verify()
        self.session.process_message("Where is my refund?")
        self.ticket.assert_not_called()

    def test_english_order_word_does_not_force_hinglish(self):
        self.assertEqual(self.session._detect_language("My order is late", "en"), "en")

    def test_generated_handover_recommendation_calls_ticket_before_confirmation(self):
        self.agent.respond_with_tools.return_value = replace(
            response("A human colleague should review this safety issue."), needs_human=True,
        )
        result = self.session.process_message("The battery is swelling.")
        self.ticket.assert_called_once()
        self.assertEqual(result.tool_response.ticket_id, "MOCK-TICKET")
        self.assertIn("create_ticket", result.tool_response.tool_calls)

    def test_short_followup_preserves_issue_topic_through_verification(self):
        self.verify()
        self.session.process_message("And how long will that take?")
        self.assertNotEqual(self.session.state.current_topic, "verification")
        self.assertEqual(self.session.state.current_topic, "delivery")


class CalendarAndEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = DataStore()

    def test_refund_calculations_respect_warehouse_closure_and_sunday(self):
        examples = (
            ("Credit card refund started on 17 October 2026", "2026-10-27"),
            ("Net banking refund started on 17 October 2026", "2026-10-24"),
            ("UPI refund started on 12 October 2026", "2026-10-15"),
            ("Net banking refund started on 12 October 2026", "2026-10-17"),
        )
        for question, expected in examples:
            with self.subTest(question=question):
                chunks = add_calendar_evidence(question, [])
                facts = calculate_policy_facts(question, chunks, self.store)
                self.assertEqual(facts["refund"]["latest_date"], expected)

    def test_unknown_payment_method_or_ambiguous_dates_are_not_guessed(self):
        question = "Refund started 12 October 2026 and 17 October 2026"
        chunks = add_calendar_evidence(question, [])
        self.assertNotIn("refund", calculate_policy_facts(question, chunks, self.store))

    def test_festive_and_historical_return_windows_come_from_current_policy(self):
        order = {
            "delivered_on": "2026-10-14", "placed_at": "2026-10-10T10:00:00",
            "items": [{"sku": "FAKE-0001", "title": "Jeans", "return_category": "fashion"}],
        }
        chunks = add_calendar_evidence("Can I return my jeans?", [], verified_order=order)
        facts = calculate_policy_facts("return", chunks, self.store, order)
        self.assertEqual(facts["return_windows"][0]["last_return_date"], "2026-11-13")
        order.update({"delivered_on": "2026-07-30"})
        order["items"][0]["return_category"] = "electronics"
        facts = calculate_policy_facts("return", chunks, self.store, order)
        self.assertEqual(facts["return_windows"][0]["last_return_date"], "2026-08-09")

    def test_nonreturnable_item_has_no_calendar_deadline(self):
        order = {
            "delivered_on": "2026-10-14", "items": [{"return_category": "innerwear"}],
        }
        chunks = add_calendar_evidence("return", [], verified_order=order)
        facts = calculate_policy_facts("return", chunks, self.store, order)
        self.assertIsNone(facts["return_windows"][0]["last_return_date"])

    def test_sources_include_text_and_distinguish_model_citations(self):
        chunks = [
            {"section_id": "RET-2", "text": "Evidence one"},
            {"section_id": "RET-4", "text": "Evidence two"},
        ]
        sources = SupportAgent._export_sources(chunks, "Supported reply.\nSources: RET-4")
        self.assertEqual(sources[0]["text"], "Evidence one")
        self.assertFalse(sources[0]["cited_in_answer"])
        self.assertTrue(sources[1]["cited_in_answer"])

    def test_rag_prompt_receives_calculated_dates_without_padding_ranked_results(self):
        model = Mock()
        model.generate.return_value = "By 2026-10-24. Sources: REF-2, KB-00.3"
        retriever = Mock()
        retriever.search.return_value = [
            {"section_id": "DEL-1", "status": "current", "text": "Delivery", "score": 0.4}
        ]
        agent = SupportAgent(llm_client=model, retriever=retriever)
        result = agent.respond_with_rag("Net banking refund started 17 October 2026")
        self.assertIn('"latest_date": "2026-10-24"', model.generate.call_args.kwargs["user_message"])
        self.assertEqual(len(result.retrieved_sources), 1)
        self.assertGreater(len(result.sources), 1)
        self.assertTrue(any(s.get("evidence_role") == "supporting_rule" for s in result.sources))


class ArtifactAndSpeechTests(unittest.TestCase):
    def test_report_reads_voice_summary_and_updates_saved_comparison(self):
        from src import build_report

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluations = root / "evaluations"
            evaluations.mkdir()
            metrics = {
                "passed_questions": 28, "total_questions": 52,
                "pass_rate": 28 / 52, "grounded_answer_accuracy": 0.63,
            }
            def load(name, directory=None):
                if name == "batch_summary.json":
                    self.assertEqual(directory, root / "sample_outputs")
                    return {"succeeded": 8, "failed": 0, "total_files": 8}
                return metrics
            with patch.object(build_report, "ROOT", root), patch.object(
                build_report, "EVALUATIONS", evaluations
            ), patch.object(build_report, "_load_json", side_effect=load):
                report = build_report.build_report()
            self.assertTrue(report.read_bytes().startswith(b"%PDF-1.4"))
            self.assertIn(b"8 succeeded, 0 failed", report.read_bytes())
            comparison = json.loads((evaluations / "step1_vs_step2.json").read_text())
            self.assertEqual(comparison["step2"]["grounded_answer_accuracy"], 0.63)

    def test_empty_transcript_gets_spoken_recovery_without_agent_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "silence.wav"
            target = directory / "response.wav"
            sf.write(source, [0.0] * 1600, 16000)
            session = Mock()
            stt = Mock()
            stt.transcribe.return_value = {"transcript": " "}
            tts = Mock()
            tts.synthesize.side_effect = lambda *args, **kwargs: sf.write(
                kwargs["output_file"], [0.0] * 1600, 16000
            )
            voice = VoiceService(session=session, stt_client=stt, tts_client=tts)
            result = voice.process_audio(source, target)
            session.process_message.assert_not_called()
            self.assertIn("repeat", result.reply)
            self.assertTrue(target.is_file())

    def test_speech_id_grouping_preserves_zeroes_and_removes_citations(self):
        result = VoiceService.normalize_for_speech(
            "Order **00123456** arrives on 2026-10-27.\nSources: DEL-1"
        )
        self.assertIn("00 12 34 56", result)
        self.assertIn("2026-10-27", result)
        self.assertNotIn("Sources", result)

    def test_batch_trace_keeps_confirmed_ticket_and_cited_sources(self):
        import run

        source = {"section_id": "ESC-2", "text": "Create ticket first", "cited_in_answer": True}
        tool = replace(
            response("Ticket created"), sources=[source], ticket_id="MOCK-TICKET",
            tool_calls=["create_ticket"], needs_human=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = SimpleNamespace(
                transcript="Human please", reply="Ticket created", output_path="response.wav", tool_response=tool,
            )
            run._save_success(directory, Path("call.wav"), result)
            trace = json.loads((directory / "trace.json").read_text())
            self.assertEqual(trace["ticket_id"], "MOCK-TICKET")
            self.assertEqual(trace["tool_calls"], ["create_ticket"])
            self.assertEqual(trace["cited_section_ids"], ["ESC-2"])
            self.assertIn("Create ticket first", (directory / "sources.json").read_text())


if __name__ == "__main__":
    unittest.main()
