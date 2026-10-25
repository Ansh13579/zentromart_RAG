
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "").strip().casefold()
        if not self.provider:
            if os.getenv("IFM_API_KEY"):
                self.provider = "ifm"
            elif os.getenv("GEMINI_API_KEY"):
                self.provider = "gemini"
            else:
                self.provider = "nvidia"

        self.extra_body = None
        self.top_p = None
        if self.provider == "ifm":
            token = os.getenv("IFM_API_KEY")
            if not token:
                raise ValueError("IFM_API_KEY is missing from your .env file.")
            self.model = os.getenv(
                "IFM_MODEL", "IFM/K2-Horizon-375B-A23B"
            )
            base_url = os.getenv("IFM_BASE_URL", "https://api.ifm.ai/v1")
            self.max_tokens = int(os.getenv("IFM_MAX_TOKENS", "8192"))
            self.max_retries = int(os.getenv("IFM_MAX_RETRIES", "3"))
            self.timeout = float(os.getenv("IFM_TIMEOUT_SECONDS", "120"))
            self.enable_thinking = None
            self.reasoning_effort = ""
            ifm_reasoning_effort = os.getenv(
                "IFM_REASONING_EFFORT", "high"
            ).strip()
            if ifm_reasoning_effort:
                self.extra_body = {
                    "chat_template_kwargs": {
                        "reasoning_effort": ifm_reasoning_effort
                    }
                }
            self.temperature = 1.0
            self.top_p = 0.95
        elif self.provider == "gemini":
            token = os.getenv("GEMINI_API_KEY")
            if not token:
                raise ValueError("GEMINI_API_KEY is missing from your .env file.")
            self.model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
            base_url = os.getenv(
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self.max_tokens = int(os.getenv("GEMINI_MAX_TOKENS", "1024"))
            self.max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
            self.timeout = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "60"))
            self.enable_thinking = None
            self.reasoning_effort = os.getenv(
                "GEMINI_REASONING_EFFORT", "low"
            ).strip()
            self.temperature = 0.3
        elif self.provider == "nvidia":
            token = os.getenv("NVIDIA_API_KEY")
            if not token:
                raise ValueError("NVIDIA_API_KEY is missing from your .env file.")
            self.model = os.getenv(
                "NVIDIA_MODEL",
                "nvidia/nemotron-3.5-super-vl-120b-a12b",
            )
            base_url = os.getenv(
                "NVIDIA_BASE_URL",
                "https://integrate.api.nvidia.com/v1",
            )
            self.max_tokens = int(os.getenv("NVIDIA_MAX_TOKENS", "1200"))
            self.max_retries = int(os.getenv("NVIDIA_MAX_RETRIES", "2"))
            self.timeout = float(os.getenv("NVIDIA_TIMEOUT_SECONDS", "60"))
            thinking_value = os.getenv(
                "NVIDIA_ENABLE_THINKING", ""
            ).strip().casefold()
            self.enable_thinking = (
                None
                if not thinking_value
                else thinking_value in {"1", "true", "yes", "on"}
            )
            self.reasoning_effort = ""
            self.temperature = 0.3
        else:
            raise ValueError(
                "Unsupported LLM_PROVIDER. Choose 'ifm', 'gemini' or 'nvidia'."
            )

        self.max_retries = max(1, self.max_retries)

        self.client = OpenAI(
            base_url=base_url,
            api_key=token,
            timeout=self.timeout,
        )

    def generate(self, system_message, user_message):
        max_tokens = self.max_tokens
        for attempt in range(self.max_retries):
            try:
                return self._generate_once(
                    system_message,
                    user_message,
                    max_tokens=max_tokens,
                )
            except (APITimeoutError, APIConnectionError, APIError) as error:
                status_code = getattr(error, "status_code", None)
                retryable = (
                    status_code is None
                    or status_code == 429
                    or status_code >= 500
                )
                if attempt + 1 < self.max_retries and retryable:
                    delay = self._retry_delay(error, attempt)
                    print(
                        f"{self.provider.title()} generation retry "
                        f"{attempt + 1}/{self.max_retries - 1} after "
                        f"{type(error).__name__} "
                        f"(status={status_code}, retry_in={delay}s).",
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"{self.provider.title()} generation failed after "
                    f"{attempt + 1} attempt(s): {error}"
                ) from error
            except RuntimeError as error:
                if (
                    "finish_reason='length'" in str(error)
                    and attempt + 1 < self.max_retries
                ):
                    max_tokens *= 2
                    print(
                        f"{self.provider.title()} response reached its token "
                        "limit; retrying "
                        f"with max_tokens={max_tokens}.",
                        flush=True,
                    )
                    continue
                raise

        raise RuntimeError(
            "The language model exhausted its response budget without text."
        )

    @staticmethod
    def _retry_delay(error, attempt):
        """Honor Retry-After when provided; otherwise use bounded backoff."""

        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return max(0.0, min(float(retry_after), 60.0))
            except (TypeError, ValueError):
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
                    return max(0.0, min(seconds, 60.0))
                except (TypeError, ValueError, OverflowError):
                    pass
        return min(2**attempt, 30)

    def _generate_once(self, system_message, user_message, *, max_tokens):
        """Make one request and validate the returned text."""

        request_kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            request_kwargs["top_p"] = self.top_p
        if self.reasoning_effort:
            request_kwargs["reasoning_effort"] = self.reasoning_effort
        if self.extra_body is not None:
            request_kwargs["extra_body"] = self.extra_body
        if self.enable_thinking is not None:
            request_kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking,
                }
            }

        response = self.client.chat.completions.create(
            **request_kwargs,
        )
        if not response.choices:
            raise RuntimeError("The language model returned no choices.")
        choice = response.choices[0]
        content = choice.message.content
        if not content:
            raise RuntimeError(
                "The language model returned no text "
                f"(finish_reason={choice.finish_reason!r}, "
                f"max_tokens={max_tokens})."
            )
        return content


if __name__ == "__main__":
    llm = LLMClient()

    result = llm.generate(
        system_message="You are a helpful AI assistant.",
        user_message="Explain machine learning in simple terms.",
    )

    if result is not None:
        print("Generated Response:")
        print(result)
