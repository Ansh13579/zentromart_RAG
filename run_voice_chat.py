"""Record and play a multi-turn ZentroMart voice conversation locally."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Any
from uuid import uuid4

import soundfile as sf

from src.agent import SupportAgent
from src.conversation import ConversationSession
from src.voice import VoiceService


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Record short microphone turns, keep one conversation session, "
            "and play each Gnani-generated spoken reply."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sample_outputs/live_sessions"),
        help="Directory where this session's turn artifacts are saved.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=8.0,
        help="Maximum microphone recording time for each turn (1 to 60 seconds).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate in Hz.",
    )
    args = parser.parse_args()
    if not 1 <= args.seconds <= 60:
        parser.error("--seconds must be between 1 and 60.")
    if args.sample_rate < 8000:
        parser.error("--sample-rate must be at least 8000 Hz.")

    try:
        import sounddevice as sd
    except ImportError as error:
        raise SystemExit(
            "Microphone chat requires the sounddevice package. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from error

    session_dir = args.output_dir / (
        datetime.now().strftime("session_%Y%m%d_%H%M%S_") + uuid4().hex[:8]
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    session = ConversationSession(agent=SupportAgent())
    voice = VoiceService(session=session)
    turn_number = 0

    print("Voice chat ready. Speak after recording starts; enter q to quit.")
    try:
        while True:
            command = input("Press Enter to record the next turn, or q to quit: ")
            if command.strip().casefold() in {"q", "quit", "exit"}:
                break

            turn_number += 1
            print(f"Recording for up to {args.seconds:g} seconds...")
            audio = sd.rec(
                int(args.seconds * args.sample_rate),
                samplerate=args.sample_rate,
                channels=1,
                dtype="int16",
            )
            sd.wait()

            turn_dir = session_dir / f"turn_{turn_number:03d}"
            turn_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
                input_path = Path(temp.name)
            try:
                sf.write(input_path, audio, args.sample_rate, subtype="PCM_16")
                result = voice.process_audio(
                    input_path,
                    output_path=turn_dir / "response.wav",
                )
            finally:
                input_path.unlink(missing_ok=True)

            response = result.tool_response
            sources = response.sources if response is not None else []
            (turn_dir / "transcript.txt").write_text(
                result.transcript,
                encoding="utf-8",
            )
            (turn_dir / "response.txt").write_text(result.reply, encoding="utf-8")
            (turn_dir / "sources.json").write_text(
                json.dumps(sources, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
            trace = {
                "turn": turn_number,
                "transcript": result.transcript,
                "response": result.reply,
                "output_audio": Path(result.output_path).name if result.output_path else None,
                "sources": sources,
                "cited_section_ids": sorted({
                    source["section_id"] for source in sources
                    if source.get("section_id") and source.get("cited_in_answer")
                }),
                "section_ids": sorted(
                    {
                        str(source["section_id"])
                        for source in sources
                        if source.get("section_id")
                    }
                ),
                "tool_calls": response.tool_calls if response is not None else [],
                "tool_results": response.tool_results if response is not None else {},
                "ticket_id": response.ticket_id if response is not None else None,
                "needs_human": response.needs_human if response is not None else False,
                "conversation_state": session.state.public_snapshot(),
            }
            (turn_dir / "trace.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )

            print(f"You: {result.transcript or '[no speech recognized]'}")
            print(f"Agent: {result.reply}")
            if result.output_path:
                reply_audio, reply_rate = sf.read(result.output_path, dtype="float32")
                sd.play(reply_audio, reply_rate)
                sd.wait()
            print(f"Turn artifacts: {turn_dir}")
    except KeyboardInterrupt:
        print("\nVoice chat stopped.")
    finally:
        print(f"Session artifacts: {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
