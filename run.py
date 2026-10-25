"""Process a folder of customer audio calls through the ZentroMart agent."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from src.agent import SupportAgent
from src.conversation import ConversationSession
from src.voice import VoiceService


_AUDIO_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
}


def _json_default(value: Any) -> Any:
    """Serialize common date and path objects returned by tools."""

    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _call_directory_name(index: int, audio_path: Path) -> str:
    """Build a stable, filesystem-safe per-call directory name."""

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", audio_path.stem).strip("._-")
    return f"call_{index:03d}" + (f"_{stem}" if stem else "")


def _save_success(call_dir: Path, audio_path: Path, result: Any) -> None:
    """Persist one call's transcript, reply, speech and evidence trace."""

    (call_dir / "transcript.txt").write_text(result.transcript, encoding="utf-8")
    (call_dir / "response.txt").write_text(result.reply, encoding="utf-8")

    tool_response = result.tool_response
    sources = tool_response.sources if tool_response is not None else []
    tool_results = tool_response.tool_results if tool_response is not None else {}
    trace = {
        "status": "success",
        "input_file": audio_path.name,
        "transcript": result.transcript,
        "response": result.reply,
        "output_audio": (
            Path(result.output_path).name if result.output_path else None
        ),
        "sources": sources,
        "cited_section_ids": sorted({
            str(source["section_id"]) for source in sources
            if source.get("section_id") and source.get("cited_in_answer")
        }),
        "retrieved_section_ids": [
            source["section_id"] for source in sources
            if source.get("evidence_role", "retrieved") == "retrieved"
            and source.get("section_id")
        ],
        "section_ids": sorted(
            {
                str(source["section_id"])
                for source in sources
                if source.get("section_id")
            }
        ),
        "retrieval_scores": (
            tool_response.retrieval_scores if tool_response is not None else []
        ),
        "tool_calls": tool_response.tool_calls if tool_response is not None else [],
        "tool_results": tool_results,
        "ticket_id": tool_response.ticket_id if tool_response is not None else None,
        "state_changes": (
            tool_response.state_changes if tool_response is not None else {}
        ),
        "needs_human": (
            tool_response.needs_human if tool_response is not None else False
        ),
        "abstained": (
            tool_response.abstained if tool_response is not None else False
        ),
    }
    _write_json(call_dir / "sources.json", sources)
    _write_json(call_dir / "trace.json", trace)


def _save_failure(call_dir: Path, audio_path: Path, error: Exception) -> dict[str, str]:
    message = f"{type(error).__name__}: {error}"
    (call_dir / "error.txt").write_text(message, encoding="utf-8")
    _write_json(
        call_dir / "trace.json",
        {
            "status": "failed",
            "input_file": audio_path.name,
            "error": message,
        },
    )
    return {"input_file": audio_path.name, "error": message}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Process each audio file in a folder and save a transcript, "
            "ZentroMart reply, audio reply, sources and tool trace per call."
        )
    )
    parser.add_argument(
        "--input_folder",
        "--input-folder",
        required=True,
        type=Path,
        help="Folder containing customer audio files.",
    )
    parser.add_argument(
        "--output_folder",
        "--output-folder",
        required=True,
        type=Path,
        help="Folder where per-call artifacts and batch_summary.json are saved.",
    )
    args = parser.parse_args(argv)

    input_dir = args.input_folder.expanduser().resolve()
    output_dir = args.output_folder.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"Input folder does not exist or is not a directory: {input_dir}")

    audio_files = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in _AUDIO_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )
    if not audio_files:
        parser.error(
            f"No supported audio files found in {input_dir}. "
            f"Supported extensions: {', '.join(sorted(_AUDIO_EXTENSIONS))}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    completed: list[dict[str, str]] = []

    # Reuse the retrieval and LLM clients, while keeping each audio file as an
    # independent customer call with its own conversation history.
    shared_agent: SupportAgent | None = None
    agent_error: Exception | None = None
    try:
        shared_agent = SupportAgent()
    except Exception as error:  # Report initialization failures for every call.
        agent_error = error

    for index, audio_path in enumerate(audio_files, start=1):
        call_dir = output_dir / _call_directory_name(index, audio_path)
        call_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(audio_files)}] Processing {audio_path.name}...")
        try:
            if agent_error is not None:
                raise RuntimeError(f"Could not initialize support agent: {agent_error}")
            session = ConversationSession(agent=shared_agent)
            service = VoiceService(session=session)
            result = service.process_audio(
                audio_path,
                output_path=call_dir / "response.wav",
            )
            _save_success(call_dir, audio_path, result)
            completed.append(
                {
                    "input_file": audio_path.name,
                    "output_directory": call_dir.name,
                    "transcript_nonempty": bool(result.transcript.strip()),
                    "empty_transcript_recovered": not bool(result.transcript.strip()),
                    "speech_reply_created": bool(
                        result.output_path and Path(result.output_path).is_file()
                    ),
                }
            )
            print(f"  Saved artifacts to {call_dir}")
        except Exception as error:
            failures.append(_save_failure(call_dir, audio_path, error))
            print(f"  FAILED: {type(error).__name__}: {error}", file=sys.stderr)

    summary = {
        "input_folder": str(input_dir),
        "output_folder": str(output_dir),
        "total_files": len(audio_files),
        "succeeded": len(completed),
        "failed": len(failures),
        "voice_metrics": {
            "complete_stt_tts_pipeline": len(completed),
            "tts_audio_created": sum(
                bool(call["speech_reply_created"]) for call in completed
            ),
            "empty_transcripts_recovered": sum(
                bool(call["empty_transcript_recovered"]) for call in completed
            ),
            "failed_files": len(failures),
        },
        "calls": completed,
        "failures": failures,
    }
    _write_json(output_dir / "batch_summary.json", summary)
    print(
        f"Finished: {len(completed)} succeeded, {len(failures)} failed. "
        f"Summary: {output_dir / 'batch_summary.json'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
