# Rules.md — AI & Engineering Boundaries

## 1. Absolute Constraints (Never Violate)
1. **Fully Local & Zero-Cloud**: Never import or invoke cloud AI SDKs (OpenAI, Anthropic, ElevenLabs, Deepgram, Groq, Google GenAI, etc.). All inference must stay strictly local.
2. **Deterministic Grounding**:
   - The LLM temperature must remain pinned at `0.0`.
   - Never answer user questions about accounts or banking products using parametric knowledge. Answers MUST derive solely from retrieved KB text.
   - If retrieval confidence falls below `min_score` (default 0.45) or returns empty, the assistant must return the standard refusal phrase immediately.
3. **Strict Numeric Verification**:
   - Spoken responses must not invent or approximate numbers, rates, tenure periods, or fee figures.
   - Every number in the generated response must exist verbatim in the retrieved source context chunks.
4. **Single Source of Configuration**:
   - Never hardcode file paths, timeouts, model names, or thresholds across submodules.
   - All settings must reside in `bank_voice_assistant.config.Settings` and be overridable via `BVA_*` environment variables.
5. **Clean Exception Handling**:
   - All expected runtime failures must subclass `BankVoiceAssistantError` in `errors.py`.
   - The CLI and voice loop must catch these errors and present clean, actionable diagnostics to the user, never unformatted Python stack traces.

---

## 2. Coding & Architectural Conventions
1. **Packaging & Isolation**:
   - All source code belongs under `src/bank_voice_assistant/`.
   - Only `transport/` and audio scripts may import `sounddevice` or access hardware mic devices.
2. **Testing Integrity**:
   - Unit tests must be fast (< 2 seconds total) and completely deterministic.
   - Unit tests must **never** initialize hardware microphones, invoke local Ollama models, or download ONNX weights during execution. Use mocks, fakes, or monkeypatching.
3. **Repository Hygiene**:
   - Never stage or commit model binaries (`*.onnx`), audio recordings (`*.wav`), pip caches (`.cache/`), or virtual environments (`test/`).
   - `.gitignore` must strictly exclude all large or non-deterministic binary assets.
4. **Documentation Sync**:
   - Keep `Memory.md` continuously updated at the end of every active coding phase or state change.
