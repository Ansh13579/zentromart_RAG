"""Gnani voice wrapper for the stateful ZentroMart support agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Protocol
from uuid import uuid4

import soundfile as sf

if __package__:
    from .config import (
        GNANI_API_KEY,
        GNANI_MAX_AUDIO_DURATION_SECONDS,
        GNANI_STT_LANGUAGE,
        GNANI_TTS_LANGUAGE,
        GNANI_TTS_MODEL,
        GNANI_TTS_VOICE,
        VOICE_OUTPUT_DIR,
    )
    from .conversation import ConversationSession
else:
    from config import (
        GNANI_API_KEY,
        GNANI_MAX_AUDIO_DURATION_SECONDS,
        GNANI_STT_LANGUAGE,
        GNANI_TTS_LANGUAGE,
        GNANI_TTS_MODEL,
        GNANI_TTS_VOICE,
        VOICE_OUTPUT_DIR,
    )
    from conversation import ConversationSession

try:
    from gnani.stt import GnaniSTTClient
    from gnani.tts import GnaniTTSClient
except ImportError as error:  # pragma: no cover - depends on optional package.
    GnaniSTTClient = None  # type: ignore[assignment,misc]
    GnaniTTSClient = None  # type: ignore[assignment,misc]
    _GNANI_IMPORT_ERROR = error
else:
    _GNANI_IMPORT_ERROR = None


class STTClient(Protocol):
    def transcribe(
        self,
        audio: str | Path,
        language_code: str = "en-IN",
        *,
        format: str = "transcribe",
    ) -> dict[str, Any]:
        """Transcribe an audio file."""


class TTSClient(Protocol):
    def synthesize(
        self,
        text: str,
        voice: str | None = "Pranav",
        *,
        model: str = "timbre-v2.5",
        language: str | None = None,
        output_file: str | Path | None = None,
    ) -> bytes:
        """Synthesize text to audio."""


@dataclass(frozen=True)
class VoiceResponse:
    """Transcript, text reply and generated audio path."""

    transcript: str
    reply: str
    output_path: str | None
    tool_response: Any | None = None


class VoiceService:
    """Orchestrate STT, the existing conversation agent and TTS."""

    def __init__(
        self,
        *,
        session: ConversationSession | None = None,
        stt_client: STTClient | None = None,
        tts_client: TTSClient | None = None,
        stt_language: str | None = None,
        tts_model: str | None = None,
        tts_voice: str | None = None,
        tts_language: str | None = None,
    ) -> None:
        if stt_client is None or tts_client is None:
            if _GNANI_IMPORT_ERROR is not None:
                raise ImportError(
                    "The Gnani SDK is required for voice support."
                ) from _GNANI_IMPORT_ERROR
            if not GNANI_API_KEY:
                raise ValueError("GNANI_API_KEY is missing from .env.")

        self.session = session or ConversationSession()
        self.stt_client = stt_client or GnaniSTTClient(
            api_key=GNANI_API_KEY
        )
        self.tts_client = tts_client or GnaniTTSClient(
            api_key=GNANI_API_KEY
        )
        self.stt_language = stt_language or GNANI_STT_LANGUAGE
        self.tts_model = tts_model or GNANI_TTS_MODEL
        self.tts_voice = tts_voice or GNANI_TTS_VOICE
        self.tts_language = tts_language or GNANI_TTS_LANGUAGE
        self.max_audio_duration_seconds = GNANI_MAX_AUDIO_DURATION_SECONDS

    def process_audio(
        self,
        audio_path: str | Path,
        output_path: str | Path | None = None,
    ) -> VoiceResponse:
        """Transcribe, process one conversation turn and synthesize the reply."""

        source = Path(audio_path)
        if not source.is_file():
            raise FileNotFoundError(f"Audio input was not found: {source}")
        self._validate_audio_duration(source)

        transcription = self.stt_client.transcribe(
            source,
            language_code=self.stt_language,
            format="transcribe",
        )
        transcript = self._extract_transcript(transcription)
        if not transcript:
            reply = (
                "I could not clearly understand the audio. "
                "Please repeat your request."
            )
            target = Path(output_path) if output_path else (
                VOICE_OUTPUT_DIR / f"reply-{uuid4().hex}.wav"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            self.tts_client.synthesize(
                self.normalize_for_speech(reply),
                voice=self.tts_voice,
                model=self.tts_model,
                language=self.tts_language,
                output_file=target,
            )
            if not target.is_file():
                raise RuntimeError(
                    "Gnani TTS did not create the expected audio file: "
                    f"{target}"
                )
            return VoiceResponse(
                transcript="",
                reply=reply,
                output_path=str(target),
            )

        conversation_response = self.session.process_message(transcript)
        reply = conversation_response.answer.strip()
        if not reply:
            raise RuntimeError("The conversation agent returned an empty reply.")

        speech_text = self.normalize_for_speech(reply)
        target = Path(output_path) if output_path else (
            VOICE_OUTPUT_DIR / f"reply-{uuid4().hex}.wav"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        self.tts_client.synthesize(
            speech_text,
            voice=self.tts_voice,
            model=self.tts_model,
            language=self.tts_language,
            output_file=target,
        )
        if not target.is_file():
            raise RuntimeError(
                f"Gnani TTS did not create the expected audio file: {target}"
            )
        return VoiceResponse(
            transcript=transcript,
            reply=reply,
            output_path=str(target),
            tool_response=conversation_response.tool_response,
        )

    def _validate_audio_duration(self, audio_path: Path) -> None:
        """Reject files longer than the Gnani REST STT limit."""

        try:
            audio_info = sf.info(audio_path)
        except RuntimeError as error:
            raise ValueError(
                f"Could not read audio metadata for {audio_path}: {error}"
            ) from error
        if audio_info.duration <= 0:
            raise ValueError("Audio input must contain at least one sample.")
        if audio_info.duration > self.max_audio_duration_seconds:
            raise ValueError(
                f"Audio duration is {audio_info.duration:.2f}s, exceeding the "
                f"configured limit of {self.max_audio_duration_seconds:.2f}s."
            )

    @staticmethod
    def _extract_transcript(result: dict[str, Any]) -> str:
        """Extract and validate the transcript returned by Gnani STT."""

        if not isinstance(result, dict):
            raise ValueError("Gnani STT returned an invalid response.")
        transcript = result.get("transcript")
        if transcript is None:
            transcript = result.get("text")
        if not isinstance(transcript, str):
            raise ValueError("Gnani STT response did not contain a transcript.")
        return transcript.strip()

    @staticmethod
    def normalize_for_speech(text: str) -> str:
        """Remove visual formatting and source citations before TTS."""

        normalized = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        normalized = re.sub(r"[*_`#]", "", normalized)
        normalized = re.sub(
            r"\s*Sources?:.*$",
            "",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
        # Read long numeric order IDs as digit groups, preserving leading zeros.
        normalized = re.sub(
            r"(?<!\d)(\d{8})(?!\d)",
            lambda match: " ".join(
                match[1][offset:offset + 2] for offset in range(0, 8, 2)
            ),
            normalized,
        )
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()


__all__ = ["VoiceResponse", "VoiceService"]
