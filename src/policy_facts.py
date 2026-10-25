"""Derive calendar facts from current policy evidence and trusted records.

No practice answers or policy amounts are stored here. Day ranges, category
windows and historical exceptions are read from the supplied policy sections.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
import re
from typing import Any

if __package__:
    from .data_store import DataStore
    from .knowledge_base import DEFAULT_KB_DIR, load_knowledge_base
else:
    from data_store import DataStore
    from knowledge_base import DEFAULT_KB_DIR, load_knowledge_base


_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "जनवरी": 1, "फ़रवरी": 2, "फरवरी": 2, "मार्च": 3, "अप्रैल": 4,
    "मई": 5, "जून": 6, "जुलाई": 7, "अगस्त": 8, "सितंबर": 9,
    "सितम्बर": 9, "अक्टूबर": 10, "अक्तूबर": 10, "नवंबर": 11,
    "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12,
}
_MONTH_PATTERN = "|".join(_MONTHS)
_TEXT_DATE = re.compile(
    rf"\b(\d{{1,2}})\s+({_MONTH_PATTERN})\s+(\d{{4}})\b", re.IGNORECASE
)


def parse_dates(text: str) -> list[date]:
    """Read ISO and full-year English/Hindi dates without guessing a year."""

    positioned: list[tuple[int, date]] = []
    for match in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", text):
        try:
            positioned.append((match.start(), date.fromisoformat(match.group())))
        except ValueError:
            continue
    for match in _TEXT_DATE.finditer(text):
        try:
            positioned.append(
                (match.start(), date(int(match[3]), _MONTHS[match[2].casefold()], int(match[1])))
            )
        except ValueError:
            continue
    return [value for _, value in sorted(positioned)]


@lru_cache(maxsize=1)
def _current_sections() -> tuple[dict[str, Any], ...]:
    from dataclasses import asdict

    return tuple(
        asdict(chunk)
        for chunk in load_knowledge_base(DEFAULT_KB_DIR)
        if chunk.status == "current"
    )


def add_calendar_evidence(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    verified_order: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Include policy dependencies for calendar calculations, without reranking.

    These extra sections are marked as supporting rules; the existing ranked
    candidates retain their order and scores.
    """

    text = question.casefold()
    required: set[str] = set()
    if re.search(r"refund|rrn|रिफ[़]?ंड|वापसी.*पैस", text):
        required.update(("REF-2", "KB-00.2", "KB-00.3"))
    if verified_order and re.search(r"return|exchange|वापस|रिटर्न", text):
        required.update(("RET-2", "RET-3", "RET-4", "RET-8"))
    if not required:
        return chunks
    result = [dict(chunk, evidence_role="retrieved") for chunk in chunks]
    present = {chunk.get("section_id") for chunk in result}
    for chunk in _current_sections():
        if chunk["section_id"] in required and chunk["section_id"] not in present:
            result.append({**chunk, "evidence_role": "supporting_rule", "score": 0.0})
            present.add(chunk["section_id"])
    return result


def _refund_method(text: str) -> str | None:
    text = text.casefold()
    if "wallet" in text or "वॉलेट" in text:
        return "wallet"
    if re.search(r"upi|यूपीआई|यू पी आई", text):
        return "upi"
    if re.search(r"credit|debit|card|कार्ड", text):
        return "card"
    if re.search(r"net[ _-]?banking|नेट\s*बैंकिंग", text):
        return "net banking"
    if re.search(r"neft|bank[ _-]?transfer|बैंक\s*ट्रांसफर", text):
        return "bank transfer"
    return None


def _table_rows(text: str, columns: int) -> list[list[str]]:
    # KB chunking collapses whitespace; Markdown rows may share one line.
    pattern = r"\|" + r"([^|]+)\|" * columns
    return [
        [cell.strip(" `") for cell in match.groups()]
        for match in re.finditer(pattern, text)
    ]


def calculate_policy_facts(
    question: str,
    chunks: list[dict[str, Any]],
    store: DataStore,
    verified_order: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate dates from policy table values; missing inputs stay unknown."""

    sections = {
        chunk["section_id"]: chunk["text"]
        for chunk in chunks
        if chunk.get("status") == "current"
    }
    facts: dict[str, Any] = {}
    if "REF-2" in sections:
        dates = parse_dates(question)
        method = _refund_method(question)
        started = dates[0] if len(dates) == 1 else None
        if verified_order:
            for prefix in ("return", "cancellation", "rto"):
                raw = verified_order.get(f"{prefix}_refund_initiated_on")
                if raw:
                    started = started or date.fromisoformat(str(raw)[:10])
                    method = method or _refund_method(
                        str(verified_order.get(f"{prefix}_refund_method") or "")
                    )
                    break
        for cells in _table_rows(sections["REF-2"], 2):
            if len(cells) < 2 or not method or _refund_method(cells[0]) != method:
                continue
            bounds = re.search(r"(\d+)\s+to\s+(\d+)\s+business days", cells[1])
            if bounds and started and "KB-00.3" in sections:
                minimum, maximum = map(int, bounds.groups())
                latest = store.business_day_deadline(started, maximum)
                facts["refund"] = {
                    "method": cells[0], "initiated_on": started.isoformat(),
                    "business_days_min": minimum, "business_days_max": maximum,
                    "earliest_date": store.business_day_deadline(started, minimum).isoformat(),
                    "latest_date": latest.isoformat(),
                    "business_days_past_latest": max(
                        0, store.business_days_between(latest, store.reference_date)
                    ),
                    "today": store.reference_date.isoformat(),
                    "counting_rule": "Exclude initiation day, Sundays and holidays.csv dates.",
                    "source_section_ids": ["REF-2", "KB-00.3"],
                }
            break

    if verified_order and verified_order.get("delivered_on") and "RET-2" in sections:
        delivered = date.fromisoformat(str(verified_order["delivered_on"])[:10])
        windows: dict[str, int | None] = {}
        for cells in _table_rows(sections["RET-2"], 3):
            if len(cells) == 3:
                window = re.search(r"(\d+)\s+days", cells[1])
                if window:
                    windows[cells[2]] = int(window[1])
                elif "not returnable" in cells[1].casefold():
                    windows[cells[2]] = None
        returns: list[dict[str, Any]] = []
        for item in verified_order.get("items", []):
            category = str(item.get("return_category", ""))
            if category not in windows:
                continue
            window = windows[category]
            source_ids = ["RET-2"]
            legacy = re.search(r"earlier\s+(\d+)-day", sections.get("RET-3", ""))
            effective = parse_dates(sections.get("RET-3", ""))
            if category == "electronics" and legacy and effective and delivered < effective[0]:
                window = int(legacy[1])
                source_ids.append("RET-3")
            festive = re.search(
                rf"between\s+(\d{{1,2}})\s+({_MONTH_PATTERN})\s+and\s+(\d{{1,2}})\s+({_MONTH_PATTERN})\s+(\d{{4}})",
                sections.get("RET-8", ""), re.IGNORECASE,
            )
            festive_window = re.search(r"within\s+(\d+)\s+days", sections.get("RET-8", ""))
            placed = str(verified_order.get("placed_at") or "")[:10]
            if category == "fashion" and festive and festive_window and placed:
                year = int(festive[5])
                start = date(year, _MONTHS[festive[2].casefold()], int(festive[1]))
                end = date(year, _MONTHS[festive[4].casefold()], int(festive[3]))
                if start <= date.fromisoformat(placed) <= end:
                    window = int(festive_window[1])
                    source_ids.append("RET-8")
            returns.append({
                "sku": item.get("sku"), "title": item.get("title"),
                "category": category, "window_days": window,
                "delivered_on": delivered.isoformat(),
                "last_return_date": store.return_deadline(delivered, window).isoformat() if window is not None else None,
                "source_section_ids": source_ids,
                "condition_note": "Window calculation only; item condition and exceptions still apply.",
            })
        if returns:
            facts["return_windows"] = returns
    return facts
