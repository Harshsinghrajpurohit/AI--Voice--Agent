# Design.md — Conversational UX & Speech Design

## 1. Conversational Style & Voice Guidelines
- **Role**: Northwind Bank Virtual Representative.
- **Tone**: Professional, precise, reassuring, and concise.
- **Auditory Constraint**: Responses are listened to, not read on a screen.
  - Keep sentences short and direct. Avoid complex nested clauses.
  - Maximum answer length: **35 words** or **3 sentences**.
  - Speak figures cleanly (e.g., "four point five percent" or "five hundred dollars").
  - Do not recite long multi-column tables or bulleted lists over audio; summarize the key takeaway and offer assistance for details.

---

## 2. Refusal & Fallback Behavior
When an answer cannot be grounded from retrieved documents or when the query is out of scope:
- **Standard Spoken Refusal**:
  > *"I apologize, but I do not have verified records for that question. Please check our online banking portal or speak with a branch representative."*
- **No Speculation**: Never say *"I think"*, *"It might be"*, or approximate numbers.

---

## 3. Audio & Voice Characteristics
- **Voice Model**: Piper `en_US-lessac-medium` (clear, natural American English female voice).
- **Sampling Rate**:
  - Input: 16,000 Hz, 16-bit PCM Mono (optimized for WebRTC VAD and Whisper).
  - Output: 22,050 Hz (native Lessac output frequency).
- **Endpointing Tuning**:
  - Silence Detection: 800 ms of continuous silence after speech indicates the user has finished their query.
  - Energy Gate: RMS energy gate 800 to ignore soft background desk noise.
