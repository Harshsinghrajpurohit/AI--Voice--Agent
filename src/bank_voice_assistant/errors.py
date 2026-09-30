"""Project-wide exception hierarchy.

Every failure mode the assistant can hit has a dedicated type so the CLI can
turn it into an actionable message instead of a traceback.
"""

from __future__ import annotations


class BankVoiceAssistantError(Exception):
    """Base class for all project errors."""


class ConfigError(BankVoiceAssistantError):
    """Invalid or contradictory configuration."""


class KnowledgeBaseError(BankVoiceAssistantError):
    """The knowledge base is missing, empty, or malformed."""


class IndexNotBuiltError(BankVoiceAssistantError):
    """The retrieval index has not been built yet.

    Carries the exact command the user should run next.
    """

    def __init__(self, index_dir: object, build_command: str = "python -m bank_voice_assistant --build-index") -> None:
        self.index_dir = index_dir
        self.build_command = build_command
        super().__init__(f"No index found in {index_dir!s}. Build it with:\n  {build_command}")


class RetrievalError(BankVoiceAssistantError):
    """Embedding or index lookup failed."""


class AudioError(BankVoiceAssistantError):
    """Microphone capture, VAD, or playback failed."""


class TranscriptionError(BankVoiceAssistantError):
    """Speech-to-text failed."""


class SpeechSynthesisError(BankVoiceAssistantError):
    """Text-to-speech failed."""


class LLMError(BankVoiceAssistantError):
    """The local LLM server could not produce an answer."""


class GuardrailViolation(BankVoiceAssistantError):
    """An answer was rejected because it was not supported by the knowledge base."""


class EvalError(BankVoiceAssistantError):
    """The golden evaluation dataset is missing, malformed, or inconsistent."""
