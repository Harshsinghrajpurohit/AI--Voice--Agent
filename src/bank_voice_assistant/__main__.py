"""Make ``python -m bank_voice_assistant`` behave like the ``bank-voice`` script."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
