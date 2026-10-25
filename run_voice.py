"""Run one audio turn and save the voice-phase artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.voice import VoiceService


def _json_default(value: Any) -> Any:
    """Serialize values commonly returned by the offline support tools."""

    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe one audio file and generate a ZentroMart voice reply."
    )
    parser.add_argument("audio", type=Path, help="Input audio file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sample_outputs"),
        help="Directory for transcript, response and trace artifacts.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    response_audio = output_dir / "response.wav"

    service = VoiceService()
    result = service.process_audio(args.audio, response_audio)

    (output_dir / "transcript.txt").write_text(
        result.transcript,
        encoding="utf-8",
    )
    (output_dir / "response.txt").write_text(
        result.reply,
        encoding="utf-8",
    )

    tool_response = result.tool_response
    sources = tool_response.sources if tool_response else []
    trace = {
        "output_path": result.output_path,
        "cited_section_ids": sorted({
            source["section_id"] for source in sources
            if source.get("section_id") and source.get("cited_in_answer")
        }),
        "retrieval_scores": (
            tool_response.retrieval_scores if tool_response else []
        ),
        "tool_results": tool_response.tool_results if tool_response else {},
        "tool_calls": tool_response.tool_calls if tool_response else [],
        "ticket_id": tool_response.ticket_id if tool_response else None,
        "state_changes": tool_response.state_changes if tool_response else {},
        "needs_human": tool_response.needs_human if tool_response else False,
        "abstained": tool_response.abstained if tool_response else False,
    }
    (output_dir / "sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output_dir / "trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    print(f"Transcript saved to {output_dir / 'transcript.txt'}")
    print(f"Response saved to {output_dir / 'response.txt'}")
    if result.output_path:
        print(f"Audio saved to {result.output_path}")
    print(f"Sources saved to {output_dir / 'sources.json'}")
    print(f"Trace saved to {output_dir / 'trace.json'}")


if __name__ == "__main__":
    main()
