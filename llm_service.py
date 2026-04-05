import os
import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError, APIConnectionError, AuthenticationError, RateLimitError, APIStatusError

load_dotenv()

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL       = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS  = 1200
DEFAULT_TIMEOUT     = 30.0  # seconds


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class LLMResult:
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None

    @property
    def failed(self) -> bool:
        return not self.success


# ── Service ───────────────────────────────────────────────────────────────────

class LLMService:
    """Thin wrapper around the OpenAI chat API that always returns structured JSON."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file or environment."
            )

        self._client = OpenAI(
            api_key=api_key,
            http_client=httpx.Client(timeout=timeout),
        )
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens

    # ── Public ────────────────────────────────────────────────────────────────

    def analyze(self, user_input: str, system_prompt: str, max_tokens: int | None = None) -> LLMResult:
        """
        Send user_input + system_prompt to the model.
        Returns LLMResult with parsed JSON in .data, or an error description.
        Optionally override the instance-level max_tokens per call.
        """
        user_input = user_input.strip()
        if not user_input:
            return LLMResult(success=False, error="Input is empty.", error_code="EMPTY_INPUT")

        try:
            raw = self._call_api(user_input, system_prompt, max_tokens=max_tokens or self.max_tokens)
            data = self._parse_json(raw)
            return LLMResult(success=True, data=data)

        except APITimeoutError:
            msg = f"Request timed out after {DEFAULT_TIMEOUT}s. Try again."
            logger.warning(msg)
            return LLMResult(success=False, error=msg, error_code="TIMEOUT")

        except APIConnectionError as e:
            msg = f"Could not reach OpenAI: {e}"
            logger.error(msg)
            return LLMResult(success=False, error=msg, error_code="CONNECTION_ERROR")

        except AuthenticationError:
            msg = "Invalid OPENAI_API_KEY. Check your .env file."
            logger.error(msg)
            return LLMResult(success=False, error=msg, error_code="AUTH_ERROR")

        except RateLimitError:
            msg = "OpenAI rate limit hit. Wait a moment and retry."
            logger.warning(msg)
            return LLMResult(success=False, error=msg, error_code="RATE_LIMIT")

        except APIStatusError as e:
            msg = f"OpenAI API error {e.status_code}: {e.message}"
            logger.error(msg)
            return LLMResult(success=False, error=msg, error_code="API_ERROR")

        except json.JSONDecodeError as e:
            msg = f"Model returned invalid JSON: {e.msg}"
            logger.error(msg)
            return LLMResult(success=False, error=msg, error_code="INVALID_JSON")

        except ValueError as e:
            msg = str(e)
            logger.warning(msg)
            return LLMResult(success=False, error=msg, error_code="CONTENT_FILTERED")

        except Exception as e:
            msg = f"Unexpected error: {e}"
            logger.exception(msg)
            return LLMResult(success=False, error=msg, error_code="UNKNOWN")

    # ── Private ───────────────────────────────────────────────────────────────

    def _call_api(self, user_input: str, system_prompt: str, max_tokens: int | None = None) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_input},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            response_format={"type": "json_object"},
        )
        choice        = response.choices[0]
        finish_reason = choice.finish_reason
        content       = (choice.message.content or "").strip()

        if finish_reason == "content_filter" or not content:
            raise ValueError(
                "The model declined to respond — your idea may involve sensitive data or privacy concerns. "
                "Try rephrasing it in more general automation terms."
            )
        if finish_reason == "length":
            raise ValueError(
                "The model's response was cut off before the JSON could be completed "
                "(finish_reason=length). The idea may be too complex for the current "
                "token limit. Please try a more concise idea or contact support."
            )

        logger.debug("Raw LLM response (%s chars): %.300s", len(content), content)
        return content

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """
        Parse JSON from the model response.
        Strips accidental markdown fences (```json ... ```) before parsing.
        Raises json.JSONDecodeError if the content is not valid JSON.
        """
        cleaned = raw
        if cleaned.startswith("```"):
            # strip opening fence (```json or ```)
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise json.JSONDecodeError("Expected a JSON object", cleaned, 0)
        return data
