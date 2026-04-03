"""
Professional Mode text processor — OpenAI API integration.

Cleans up dictated text by fixing tone, grammar, and punctuation
via the OpenAI chat-completion API.  The API key is held **only** in
memory and is never logged, printed, or persisted to disk by this module.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from openai import AuthenticationError, OpenAI, OpenAIError

log = logging.getLogger(__name__)

_KEYRING_SERVICE = "cv2t"
_KEYRING_USERNAME = "openai_api_key"

# Timeout for API requests (connect, read) in seconds.
_REQUEST_TIMEOUT = 15.0


def _sanitize_error(exc: BaseException, api_key: str) -> str:
    """Return an error message with the API key redacted."""
    msg = str(exc)
    if api_key:
        msg = msg.replace(api_key, "***")
    return msg


def _build_system_prompt(
    fix_tone: bool, fix_grammar: bool, fix_punctuation: bool
) -> str:
    """Build the system prompt from the enabled cleanup flags."""
    rules: list[str] = []
    if fix_tone:
        rules.append(
            "Make the tone professional and neutral. Remove emotional, "
            "aggressive, or unprofessional language while preserving the "
            "original meaning and intent."
        )
    if fix_grammar:
        rules.append("Fix grammar errors.")
    if fix_punctuation:
        rules.append("Add proper punctuation and capitalization.")

    if not rules:
        # All flags off — nothing to do; caller should skip the API call.
        return ""

    numbered = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))
    return (
        "You are a text cleanup assistant. Rewrite the following dictated "
        "text with these corrections:\n"
        f"{numbered}\n\n"
        "Preserve the original meaning and intent. "
        "Output only the corrected text, nothing else."
    )


class TextProcessor:
    """Send dictated text to OpenAI for professional cleanup.

    The *api_key* is stored only as an instance attribute in memory.
    It is **never** logged, printed, or included in error messages.
    """

    def __init__(self, api_key: str, model: str = "gpt-5.4-mini") -> None:
        self._api_key = api_key
        self._model = model
        self._client: Optional[OpenAI] = None
        self._ensure_client()

    def _ensure_client(self) -> None:
        if self._client is None and self._api_key:
            self._client = OpenAI(
                api_key=self._api_key, timeout=_REQUEST_TIMEOUT
            )

    # ── Public API ────────────────────────────────────────────────────────

    def process(
        self,
        text: str,
        *,
        fix_tone: bool = True,
        fix_grammar: bool = True,
        fix_punctuation: bool = True,
    ) -> str:
        """Clean up *text* according to the enabled flags.

        Returns the cleaned text on success, or the original *text*
        unchanged on any API failure (graceful degradation).
        """
        if not text or not text.strip():
            return text

        system_prompt = _build_system_prompt(fix_tone, fix_grammar, fix_punctuation)
        if not system_prompt:
            # All cleanup flags are disabled — pass through unchanged.
            return text

        self._ensure_client()
        if self._client is None:
            log.warning("Professional Mode: no API key configured — skipping cleanup")
            return text

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
            )
            cleaned = response.choices[0].message.content
            return cleaned.strip() if cleaned else text
        except Exception as exc:
            log.error(
                "Professional Mode API error: %s",
                _sanitize_error(exc, self._api_key),
            )
            return text

    def validate_key(self) -> tuple[bool, str]:
        """Validate the API key with a lightweight API call.

        Returns ``(success, message)``.
        """
        self._ensure_client()
        if self._client is None:
            return False, "No API key provided"

        try:
            self._client.models.list()
            return True, "API key is valid"
        except AuthenticationError:
            return False, "Invalid API key"
        except OpenAIError as exc:
            return False, f"API error: {_sanitize_error(exc, self._api_key)}"
        except Exception as exc:
            return False, f"Unexpected error: {_sanitize_error(exc, self._api_key)}"


# ── Keyring helpers ──────────────────────────────────────────────────────────


def load_api_key_from_keyring() -> str:
    """Load the stored API key from Windows Credential Manager.

    Returns an empty string if *keyring* is unavailable or no key is stored.
    """
    try:
        import keyring

        value = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
        return value or ""
    except Exception:
        log.debug("Could not load API key from keyring", exc_info=True)
        return ""


def save_api_key_to_keyring(api_key: str) -> None:
    """Persist the API key to Windows Credential Manager."""
    try:
        import keyring

        keyring.set_password(_KEYRING_SERVICE, _KEYRING_USERNAME, api_key)
    except Exception:
        log.warning("Could not save API key to keyring", exc_info=True)


def delete_api_key_from_keyring() -> None:
    """Remove the stored API key from Windows Credential Manager."""
    try:
        import keyring

        keyring.delete_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
    except Exception:
        # May raise if no credential exists — that's fine.
        log.debug("Could not delete API key from keyring", exc_info=True)
