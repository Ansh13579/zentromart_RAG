"""Generate the four-page capstone report with vector graphics and no dependencies.

Run with --preview-dir report_previews to also write matching SVG page previews.
Evaluation metrics are read from saved JSON; this command never calls a model.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from html import escape
import json
from pathlib import Path
import re
import struct
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVALUATIONS = ROOT / "evaluations"
PAGE_W, PAGE_H = 595.28, 841.89
MARGIN = 42.0
WIDTH = PAGE_W - MARGIN * 2
NAVY = "142D40"
TEAL = "087E83"
INK = "233A4B"
MUTED = "617383"
PALE = "F1F5F7"
LINE = "DCE5E9"
AMBER = "986018"
AMBER_BG = "FFF6E7"
WHITE = "FFFFFF"

# Standard PDF Helvetica metrics, in 1/1000 em; these keep line wrapping and
# table alignment identical between the PDF and the SVG previews.
_UPPER = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", (
    667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833,
    722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611,
)))
_LOWER = dict(zip("abcdefghijklmnopqrstuvwxyz", (
    556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833,
    556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500,
)))
_SPECIAL = {
    " ": 278, ".": 278, ",": 278, ":": 278, ";": 278, "-": 333,
    "/": 278, "(": 333, ")": 333, "%": 889, "+": 584, "&": 667,
    "'": 191, '"': 355, "!": 278, "?": 556, "=": 584, "_": 556,
}


def _load_json(name: str, directory: Path | None = None) -> dict[str, Any]:
    path = (directory if directory is not None else EVALUATIONS) / name
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_list(name: str) -> list[dict[str, Any]]:
    try:
        value = json.loads((EVALUATIONS / name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _ratio(metrics: dict[str, Any], numerator: str, denominator: str) -> str:
    return f"{metrics.get(numerator, '-')}/{metrics.get(denominator, '-')}"


def _clean(text: Any) -> str:
    return (
        str(text).replace("\u2019", "'").replace("\u2018", "'")
        .replace("\u2013", "-").replace("\u2014", "-")
        .encode("ascii", "replace").decode("ascii")
    )


def _text_width(text: str, size: float, font: str = "F1") -> float:
    if font == "F3":
        return len(text) * size * 0.6
    units = sum(
        _UPPER.get(ch, _LOWER.get(ch, _SPECIAL.get(ch, 556))) for ch in text
    )
    # Slightly conservative wrapping for the bold face.
    return units * size / 1000 * (1.045 if font == "F2" else 1.0)


def _wrap(text: str, width: float, size: float, font: str = "F1") -> list[str]:
    lines: list[str] = []
    line = ""
    for word in _clean(text).split():
        candidate = f"{line} {word}" if line else word
        if line and _text_width(candidate, size, font) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def _rgb(hex_color: str) -> tuple[float, float, float]:
    return tuple(int(hex_color[index:index + 2], 16) / 255 for index in (0, 2, 4))


class Page:
    """Small vector canvas with a top-left origin and matching PDF/SVG output."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.svg: list[str] = []

    def rect(self, x: float, y: float, w: float, h: float, color: str) -> None:
        red, green, blue = _rgb(color)
        self.commands.append(
            f"q {red:.4f} {green:.4f} {blue:.4f} rg {x:.2f} {PAGE_H-y-h:.2f} {w:.2f} {h:.2f} re f Q"
        )
        self.svg.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#{color}"/>'
        )

    def line(self, x: float, y: float, end_x: float, end_y: float, color: str = LINE) -> None:
        red, green, blue = _rgb(color)
        self.commands.append(
            f"q {red:.4f} {green:.4f} {blue:.4f} RG 0.7 w "
            f"{x:.2f} {PAGE_H-y:.2f} m {end_x:.2f} {PAGE_H-end_y:.2f} l S Q"
        )
        self.svg.append(
            f'<line x1="{x}" y1="{y}" x2="{end_x}" y2="{end_y}" stroke="#{color}" stroke-width="0.7"/>'
        )

    def text(
        self, x: float, y: float, value: Any, size: float = 10,
        color: str = INK, font: str = "F1", align: str = "left",
    ) -> None:
        text = _clean(value)
        if align == "right":
            x -= _text_width(text, size, font)
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        red, green, blue = _rgb(color)
        self.commands.append(
            f"BT /{font} {size:.2f} Tf {red:.4f} {green:.4f} {blue:.4f} rg "
            f"1 0 0 1 {x:.2f} {PAGE_H-y:.2f} Tm ({escaped}) Tj ET"
        )
        family = "monospace" if font == "F3" else "Arial,Helvetica,sans-serif"
        weight = "700" if font == "F2" else "400"
        self.svg.append(
            f'<text x="{x}" y="{y}" fill="#{color}" font-family="{family}" '
            f'font-size="{size}" font-weight="{weight}">{escape(text)}</text>'
        )

    def paragraph(
        self, x: float, y: float, text: str, width: float = WIDTH,
        size: float = 9.5, color: str = INK, leading: float = 14,
        font: str = "F1",
    ) -> float:
        for line in _wrap(text, width, size, font):
            self.text(x, y, line, size, color, font)
            y += leading
        return y

    def section(self, y: float, label: str) -> None:
        self.rect(MARGIN, y - 9, 3, 12, TEAL)
        self.text(MARGIN + 11, y, label, 10, NAVY, "F2")

    def card(self, x: float, y: float, w: float, value: str, label: str, detail: str) -> None:
        self.rect(x, y, w, 77, PALE)
        self.rect(x, y, 3, 77, TEAL)
        self.text(x + 14, y + 30, value, 25, NAVY, "F2")
        self.text(x + 14, y + 49, label, 8.5, INK, "F2")
        self.text(x + 14, y + 65, detail, 7.5, MUTED)

    def note(self, y: float, title: str, body: str, *, caution: bool = False, height: float = 65) -> None:
        self.rect(MARGIN, y, WIDTH, height, AMBER_BG if caution else PALE)
        color = AMBER if caution else TEAL
        self.rect(MARGIN, y, 3, height, color)
        self.text(MARGIN + 14, y + 20, title, 9, color, "F2")
        self.paragraph(MARGIN + 14, y + 37, body, WIDTH - 28, 8.6, INK, 12)

    def table(
        self, y: float, columns: list[float], headers: list[str],
        rows: list[list[str]], *, row_height: float = 25, size: float = 9,
    ) -> float:
        self.rect(MARGIN, y, WIDTH, 27, NAVY)
        x = MARGIN
        for width, label in zip(columns, headers):
            self.text(x + 11, y + 18, label, 8.2, WHITE, "F2")
            x += width
        y += 27
        for index, row in enumerate(rows):
            if index % 2 == 0:
                self.rect(MARGIN, y, WIDTH, row_height, PALE)
            x = MARGIN
            for col_index, (width, value) in enumerate(zip(columns, row)):
                self.paragraph(
                    x + 11, y + 17, value, width - 22, size,
                    TEAL if col_index == len(row) - 1 else INK,
                    12, "F2" if col_index == len(row) - 1 else "F1",
                )
                x += width
            y += row_height
            self.line(MARGIN, y, PAGE_W - MARGIN, y)
        return y

    def header(self, number: int, title: str, subtitle: str) -> None:
        self.rect(0, 0, PAGE_W, 6, TEAL)
        self.text(MARGIN, 32, "VOICEOPS 2026  /  CAPSTONE REPORT", 8.2, TEAL, "F2")
        self.text(PAGE_W - MARGIN, 32, "ZENTROMART", 8.2, MUTED, "F2", "right")
        self.line(MARGIN, 46, PAGE_W - MARGIN, 46)
        self.text(MARGIN, 85, title, 25, NAVY, "F2")
        self.paragraph(MARGIN, 108, subtitle, WIDTH, 9.3, MUTED, 13)
        self.line(MARGIN, 792, PAGE_W - MARGIN, 792)
        self.text(MARGIN, 810, "ZentroMart  |  Implementation & evaluation", 7.2, MUTED)
        self.text(PAGE_W - MARGIN, 810, f"{number:02d} / 04", 8, TEAL, "F2", "right")

    def svg_document(self) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W}" height="{PAGE_H}" '
            f'viewBox="0 0 {PAGE_W} {PAGE_H}">'
            f'<rect width="100%" height="100%" fill="white"/>{"".join(self.svg)}</svg>'
        )


def _write_pdf(pages: list[Page], target: Path) -> None:
    """Write selectable text, standard fonts, vector shapes and PDF metadata."""

    objects = [b"", b"<< /Type /Catalog /Pages 2 0 R >>", b""]
    objects.extend(
        f"<< /Type /Font /Subtype /Type1 /BaseFont /{font} >>".encode("ascii")
        for font in ("Helvetica", "Helvetica-Bold", "Courier")
    )
    page_ids: list[int] = []
    for page in pages:
        page_id = len(objects)
        page_ids.append(page_id)
        stream = "\n".join(page.commands).encode("ascii")
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
                f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> >> "
                f"/Contents {page_id + 1} 0 R >>"
            ).encode("ascii")
        )
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream + b"\nendstream"
        )
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii")
    info_id = len(objects)
    objects.append(
        b"<< /Title (ZentroMart - VoiceOps 2026 Capstone Report) "
        b"/Author (ZentroMart VoiceOps Project) /Subject (Implementation and saved evaluation results) "
        b"/Creator (VoiceOps report generator) >>"
    )
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects[1:], 1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects)}\n".encode("ascii") + b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects)} /Root 1 0 R /Info {info_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    target.write_bytes(output)


def _timestamp(metrics: dict[str, Any]) -> str:
    raw = metrics.get("evaluated_at_utc")
    if not raw:
        return "Not recorded"
    try:
        stamp = datetime.fromisoformat(str(raw)).astimezone(timezone(timedelta(hours=5, minutes=30)))
        return stamp.strftime("%d %b %Y, %H:%M IST")
    except ValueError:
        return str(raw)


def _demo_duration(path: Path) -> float | None:
    """Read an MP4 movie duration without external media dependencies."""

    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            def atoms(start: int, end: int):
                pos = start
                while pos + 8 <= end:
                    handle.seek(pos)
                    length, kind = struct.unpack(">I4s", handle.read(8))
                    header = 8
                    if length == 1:
                        length = struct.unpack(">Q", handle.read(8))[0]
                        header = 16
                    if length == 0:
                        length = end - pos
                    if length < header or pos + length > end:
                        break
                    yield kind, pos + header, pos + length
                    pos += length
            for kind, start, end in atoms(0, path.stat().st_size):
                if kind != b"moov":
                    continue
                for kind, start, end in atoms(start, end):
                    if kind != b"mvhd":
                        continue
                    handle.seek(start)
                    payload = handle.read(40)
                    pos = 20 if payload[0] == 1 else 12
                    scale = struct.unpack_from(">I", payload, pos)[0]
                    units = struct.unpack_from(">Q" if payload[0] == 1 else ">I", payload, pos + 4)[0]
                    return units / scale if scale else None
    except (OSError, struct.error, IndexError):
        return None
    return None


def _overview(step2: dict[str, Any], conv: dict[str, Any], voice: dict[str, Any]) -> Page:
    page = Page()
    page.header(1, "ZentroMart Voice Support", "A document-grounded customer-support agent for the VoiceOps 2026 capstone.")
    page.text(MARGIN, 155, "From customer speech to a grounded reply.", 16, NAVY, "F2")
    page.paragraph(
        MARGIN, 179,
        "The system combines policy retrieval, verified synthetic order tools and conversation "
        "memory. Gnani provides both speech recognition and speech synthesis; support decisions "
        "remain in the shared text agent.",
        WIDTH, 10, leading=15,
    )
    card_w = (WIDTH - 20) / 3
    page.card(MARGIN, 230, card_w, _percent(step2.get("pass_rate")), "RAG question pass rate", f"{_ratio(step2, 'passed_questions', 'total_questions')} practice questions")
    page.card(MARGIN + card_w + 10, 230, card_w, _percent(conv.get("turn_pass_rate")), "Conversation turn pass rate", f"{_ratio(conv, 'passed_turns', 'total_turns')} customer turns")
    page.card(MARGIN + (card_w + 10) * 2, 230, card_w, _ratio(voice, "succeeded", "total_files"), "Voice batch completed", "Saved end-to-end run")
    page.section(345, "ONE TURN THROUGH THE SYSTEM")
    stages = [
        ("01 / LISTEN", "Gnani STT", "Customer audio to text"),
        ("02 / REASON", "Support agent", "Retrieval, tools, context"),
        ("03 / SPEAK", "Gnani TTS", "Reply text to speech"),
        ("04 / RECORD", "Saved artifacts", "Text, audio and evidence"),
    ]
    box_w = (WIDTH - 30) / 4
    for index, (label, title, body) in enumerate(stages):
        x = MARGIN + index * (box_w + 10)
        page.rect(x, 363, box_w, 73, NAVY if index == 1 else PALE)
        color = WHITE if index == 1 else INK
        page.text(x + 10, 380, label, 7, "8DD6D3" if index == 1 else TEAL, "F2")
        page.text(x + 10, 399, title, 10, color, "F2")
        page.paragraph(x + 10, 416, body, box_w - 18, 7.3, color, 10)
        if index < 3:
            page.text(x + box_w + 2, 401, ">", 9, TEAL, "F2")
    page.section(475, "FOUR IMPLEMENTATION STAGES")
    stage_rows = [
        ["1", "LLM baseline", "API-based assistant with role, uncertainty and action limits."],
        ["2", "Grounded retrieval", "Markdown sections, embeddings, FAISS/BM25 candidates and a multilingual cross-encoder reranker."],
        ["3", "Conversation + tools", "Order verification, contextual follow-ups, topic changes, ticket creation and human handover."],
        ["4", "Voice + batch", "Gnani STT/TTS, microphone conversation client and a folder-based evaluator with per-call artifacts."],
    ]
    y = 490
    for number, title, body in stage_rows:
        page.text(MARGIN, y + 15, number, 17, TEAL, "F2")
        page.text(MARGIN + 30, y + 12, title, 10, NAVY, "F2")
        page.paragraph(MARGIN + 30, y + 28, body, WIDTH - 34, 9, leading=12)
        page.line(MARGIN, y + 52, PAGE_W - MARGIN, y + 52)
        y += 58
    page.note(729, "Dataset rules", "Synthetic data only. Reference date: 20 October 2026. Business days: Monday-Saturday, excluding warehouse holidays; return windows use calendar days.", height=54)
    return page


def _evaluation(step1: dict[str, Any], step2: dict[str, Any], conv: dict[str, Any]) -> Page:
    page = Page()
    page.header(2, "Measured performance", "Latest saved practice runs. These results describe observed outputs, not an organizer-assigned score.")
    page.section(151, "STEP 1 VS STEP 2 / QUESTION PASS RATE")
    chart_x, chart_w = MARGIN + 113, WIDTH - 192
    for y, label, metric, color in (
        (178, "LLM baseline", step1.get("pass_rate"), MUTED),
        (217, "Grounded RAG", step2.get("pass_rate"), TEAL),
    ):
        page.text(MARGIN, y + 11, label, 9.5, INK, "F2")
        page.rect(chart_x, y, chart_w, 15, PALE)
        page.rect(chart_x, y, chart_w * max(0, min(1, float(metric or 0))), 15, color)
        page.text(PAGE_W - MARGIN, y + 11, _percent(metric), 11, color, "F2", "right")
    delta = (float(step2.get("pass_rate") or 0) - float(step1.get("pass_rate") or 0)) * 100
    page.text(chart_x, 253, f"Observed improvement: {delta:+.1f} percentage points", 8.5, TEAL, "F2")
    page.table(272, [WIDTH - 174, 87, 87], ["Question metric", "Step 1", "Step 2"], [
        ["Questions passed", _ratio(step1, "passed_questions", "total_questions"), _ratio(step2, "passed_questions", "total_questions")],
        ["Decision accuracy", _percent(step1.get("decision_accuracy")), _percent(step2.get("decision_accuracy"))],
        ["Required-fact accuracy", _percent(step1.get("required_fact_accuracy")), _percent(step2.get("required_fact_accuracy"))],
        ["Grounded-answer accuracy", "--", _percent(step2.get("grounded_answer_accuracy"))],
        ["Retrieval hit rate / MRR", "--", f"{_percent(step2.get('retrieval_hit_rate'))} / {float(step2.get('mean_reciprocal_rank') or 0):.3f}"],
        ["Unsupported-question abstention", "--", _percent(step2.get("unsupported_abstention_accuracy"))],
    ], size=9)
    page.section(483, "MULTI-TURN CONVERSATIONS")
    w = (WIDTH - 12) / 2
    page.card(MARGIN, 499, w, _ratio(conv, "passed_turns", "total_turns"), "Customer turns passed", _percent(conv.get("turn_pass_rate")))
    page.card(MARGIN + w + 12, 499, w, _ratio(conv, "passed_conversations", "total_conversations"), "Complete conversations passed", _percent(conv.get("conversation_pass_rate")))
    page.table(593, [WIDTH / 2, WIDTH / 2], ["Control / behavior", "Recorded accuracy"], [
        ["Tool calls / verification sequence", f"{_percent(conv.get('tool_call_accuracy'))} / {_percent(conv.get('verification_sequence_accuracy'))}"],
        ["Clarification / escalation", f"{_percent(conv.get('clarification_accuracy'))} / {_percent(conv.get('escalation_accuracy'))}"],
        ["Ticket creation / leakage-free rate", f"{_percent(conv.get('ticket_creation_accuracy'))} / {_percent(conv.get('data_leakage_free_rate'))}"],
    ], row_height=23, size=8.8)
    page.text(MARGIN, 711, f"Step 1: {_timestamp(step1)}    |    Step 2: {_timestamp(step2)}", 7.5, MUTED)
    page.text(MARGIN, 726, f"Conversations: {_timestamp(conv)}", 7.5, MUTED)
    page.paragraph(MARGIN, 750, "A conversation passes only when all its turns pass. Runs were recorded at different times; these results do not isolate the reranker's contribution or prove generalization to unseen cases.", WIDTH, 8.5, MUTED, 12)
    return page


def _failures(step2: dict[str, Any], conv: dict[str, Any]) -> Page:
    page = Page()
    page.header(3, "Failure review & refinements", "Observed practice failures guide the next changes. Automatically suggested root causes remain hypotheses.")
    failures = _load_list("step2_failures.json")
    retrieval_misses = sum(bool(row.get("answerable")) and not row.get("retrieval_hit", False) for row in failures)
    fact_omissions = sum(bool(row.get("required_facts_missing")) for row in failures)
    card_w = (WIDTH - 20) / 3
    page.card(MARGIN, 144, card_w, str(step2.get("failed_questions", "-")), "RAG questions failed", "Latest saved question run")
    page.card(MARGIN + card_w + 10, 144, card_w, str(retrieval_misses), "Retrieval misses", "Among answerable failures")
    page.card(MARGIN + (card_w + 10) * 2, 144, card_w, str(fact_omissions), "Required-fact omissions", "Categories may overlap")
    page.section(260, "WHAT FAILED / WHAT TO IMPROVE")
    page.table(277, [118, 184, WIDTH - 302], ["Failure pattern", "Observed issue / likely cause", "Planned correction"], [
        ["Multilingual retrieval", "Hindi/Hinglish and multi-condition questions can miss the controlling section, including opened beauty-item rules.", "Inspect the candidate pool; improve language matching and include all relevant policy conditions."],
        ["Incomplete answers", "Some answers omit a specific deadline, fee, eligibility condition or refund destination.", "Use calculated policy facts and cover each requested detail with supporting evidence."],
        ["Response decisions", "Uncertainty, clarification and escalation wording can disagree with the expected decision.", "Review intent routing and the full response before changing rules; inspect evaluator false negatives."],
        ["Conversation continuity", f"{conv.get('failed_conversations', '-')} complete conversations fail at least one turn; tool correctness alone is insufficient.", "Recheck follow-ups, verification boundaries and handovers across complete conversations."],
        ["Evaluation sensitivity", "Decisions and facts use heuristics and text matching; wording and punctuation can affect scores.", "Normalize equivalent text consistently and review failures manually; preserve original agent outputs."],
    ], row_height=74, size=8.4)
    page.section(709, "RECENT IMPLEMENTATION REFINEMENTS")
    page.paragraph(MARGIN, 730, "Order switching now clears private context and re-verifies the caller. Confirmed handovers export real ticket events. Deadlines use current policy and the holiday calendar. Evidence includes full chunk text and model-cited IDs; first replies disclose AI identity.", WIDTH, 9, leading=13)
    return page


def _voice(voice: dict[str, Any]) -> Page:
    page = Page()
    page.header(4, "Voice delivery & next steps", "End-to-end operation, reproducibility and the remaining evidence needed for a complete submission.")
    calls = voice.get("calls", [])
    audio_count = voice.get("voice_metrics", {}).get("tts_audio_created")
    if audio_count is None:
        audio_count = sum(
            (ROOT / "sample_outputs" / str(call.get("output_directory", "")) / "response.wav").is_file()
            for call in calls if isinstance(call, dict)
        )
    card_w = (WIDTH - 20) / 3
    page.card(MARGIN, 144, card_w, str(voice.get("succeeded", "-")), "Files succeeded", "Saved voice batch")
    page.card(MARGIN + card_w + 10, 144, card_w, str(voice.get("failed", "-")), "Files failed", "Batch processing errors")
    page.card(MARGIN + (card_w + 10) * 2, 144, card_w, str(audio_count), "Audio replies present", "Checked output WAV files")
    page.text(MARGIN, 245, f"Batch summary: {voice.get('succeeded', '-')} succeeded, {voice.get('failed', '-')} failed.", 9.5, TEAL, "F2")
    page.section(282, "VOICE BEHAVIOR & EVIDENCE")
    page.paragraph(MARGIN, 305, "Gnani STT transcribes short customer turns; the same conversation agent generates a reply, and Gnani TTS saves the spoken output. Empty transcripts produce a spoken repeat request. Speech normalization removes visual formatting and groups eight-digit order IDs.", WIDTH, 9.5, leading=14)
    page.table(367, [139, WIDTH - 139], ["Saved per call", "Purpose"], [
        ["transcript.txt / response.txt", "Recognized customer speech and the original agent reply."],
        ["response.wav", "Gnani-generated spoken response."],
        ["sources.json / trace.json", "Evidence text, section IDs, citations, actual tool calls and ticket details."],
    ], row_height=28, size=8.4)
    page.note(497, "Verification scope", "Local regression checks exercise order routing, re-verification, tickets, calendar rules, source exports and unclear-audio recovery. A successful batch does not measure transcription accuracy, answer correctness or voice clarity.", height=72)
    duration = _demo_duration(ROOT / "demo.mp4")
    if duration is not None:
        seconds = int(round(duration))
        stamp = f"{seconds//60}:{seconds%60:02d}"
        title = f"Demo recording: {stamp} / maximum 2:00"
        body = (
            "A recording is present. Shorten it to two minutes or less while preserving the agent's actual answers. Keep a grounded question, a follow-up, and a handover or uncertainty response."
            if duration > 120 else
            "A recording is present and within the time limit. Review both voices and confirm it demonstrates a grounded question, a contextual follow-up, and a handover or uncertainty response."
        )
        page.note(582, title, body, caution=duration > 120, height=71)
    else:
        page.note(582, "Demo recording needs review", "Add or verify demo.mp4. The video must be at most two minutes and demonstrate a grounded answer, a follow-up, and a handover or honest uncertainty.", caution=True, height=71)
    page.paragraph(MARGIN, 677, "Before submission, review English, Hindi/Hinglish and unclear/noisy speech; publish the repository URL in README.md and refresh the evaluation after further changes. Keep credentials, virtual environments and the dataset copy out of the archive.", WIDTH, 8.8, leading=12)
    page.rect(MARGIN, 720, WIDTH, 58, NAVY)
    page.text(MARGIN + 12, 737, "REPRODUCIBLE RUN COMMANDS", 7, "8DD6D3", "F2")
    page.text(MARGIN + 12, 753, "python run.py --input_folder input_folder --output_folder sample_outputs", 7.2, WHITE, "F3")
    page.text(MARGIN + 12, 767, "python src/evaluation.py   |   python src/build_report.py", 7.2, WHITE, "F3")
    return page


def build_report(preview_dir: Path | None = None) -> Path:
    step1 = _load_json("step1_metrics.json")
    step2 = _load_json("step2_metrics.json")
    conversations = _load_json("conversation_metrics.json")
    voice = _load_json("batch_summary.json", ROOT / "sample_outputs")
    if step1 and step2:
        if __package__:
            from .evaluation import save_step_comparison
        else:
            from evaluation import save_step_comparison
        save_step_comparison(step1, step2, EVALUATIONS / "step1_vs_step2.json")
    pages = [
        _overview(step2, conversations, voice),
        _evaluation(step1, step2, conversations),
        _failures(step2, conversations),
        _voice(voice),
    ]
    target = ROOT / "report.pdf"
    _write_pdf(pages, target)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        for number, page in enumerate(pages, 1):
            (preview_dir / f"page_{number}.svg").write_text(page.svg_document(), encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, help="Optional SVG preview directory.")
    args = parser.parse_args()
    print(build_report(args.preview_dir))
