"""
Voice Agent — Base Version
100% local: Faster-Whisper (STT) + Ollama/Llama 3.2 (LLM) + Piper (TTS)

Pipeline: record -> transcribe -> reason (+ optional tool call) -> speak -> repeat
This is intentionally simple (fixed-duration recording, no streaming) so you can
see the whole loop clearly before we optimize it.
"""

import json
import subprocess
import time
from datetime import datetime

import numpy as np
import ollama
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from scipy.io.wavfile import read, write

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
AUDIO_IN_FILE = "audio.wav"
AUDIO_OUT_FILE = "response.wav"
PIPER_MODEL = "en_US-lessac-medium.onnx"
LLM_MODEL = "llama3.2"

# --- VAD / endpointing config ---
FRAME_MS = 30                 # webrtcvad requires 10, 20, or 30 ms frames
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
VAD_AGGRESSIVENESS = 3        # 0 (least aggressive) to 3 (most aggressive) filtering of non-speech
ENERGY_THRESHOLD = 800        # int16 RMS below this = treat as silence, no matter what VAD says
                               # (filters ambient noise/hum that VAD's pattern-matching can mistake for speech)
SILENCE_MS_TO_STOP = 800      # how much trailing silence = "user is done talking"
SILENCE_FRAMES_TO_STOP = SILENCE_MS_TO_STOP // FRAME_MS
MAX_RECORD_SECONDS = 15       # safety cap so a stuck mic doesn't record forever
MIN_SPEECH_FRAMES = 5         # ignore tiny blips (coughs, clicks) before "speech started"

# Loaded once at startup so each turn is faster
print("Loading Whisper model...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# --- Debug: show available input devices and which one will be used ---
print("\nAvailable audio input devices:")
print(sd.query_devices())
default_input = sd.query_devices(kind="input")
print(f"\nDefault input device: {default_input['name']}\n")


# ---------------------------------------------------------------------------
# Step 1: Record audio from the microphone, using VAD to detect end-of-turn
# ---------------------------------------------------------------------------
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)


def record_audio(filename: str = AUDIO_IN_FILE) -> None:
    """
    Records from the mic until the user has clearly stopped talking
    (real endpointing), instead of a fixed duration.

    State machine:
      WAITING_FOR_SPEECH -> (enough consecutive speech frames) -> IN_SPEECH
      IN_SPEECH -> (enough consecutive silent frames) -> stop
    """
    print("Listening... (speak now)")

    frames: list[bytes] = []
    speech_started = False
    consecutive_speech_frames = 0
    consecutive_silence_frames = 0
    max_frames = int(MAX_RECORD_SECONDS * 1000 / FRAME_MS)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=FRAME_SAMPLES,
    ) as stream:
        for _ in range(max_frames):
            frame, _overflowed = stream.read(FRAME_SAMPLES)
            frame_bytes = frame.tobytes()

            # Two-factor check: webrtcvad's pattern-based detection AND a
            # simple energy (loudness) gate. Ambient noise/hum can sometimes
            # look "speech-like" to VAD alone even when it's clearly too
            # quiet to be someone talking, so we require both.
            frame_energy = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
            is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE) and frame_energy > ENERGY_THRESHOLD

            if not speech_started:
                if is_speech:
                    consecutive_speech_frames += 1
                    frames.append(frame_bytes)  # keep it, might be real speech
                    if consecutive_speech_frames >= MIN_SPEECH_FRAMES:
                        speech_started = True
                else:
                    consecutive_speech_frames = 0
                    frames = []  # discard leading silence/noise blips
            else:
                frames.append(frame_bytes)
                if is_speech:
                    consecutive_silence_frames = 0
                else:
                    consecutive_silence_frames += 1
                    if consecutive_silence_frames >= SILENCE_FRAMES_TO_STOP:
                        break  # user has stopped talking -> endpoint reached

    print("Recording complete.")
    audio = np.frombuffer(b"".join(frames), dtype="int16")
    write(filename, SAMPLE_RATE, audio)

    # --- Debug info: helps diagnose "empty transcription" issues ---
    duration_sec = len(audio) / SAMPLE_RATE
    max_amplitude = np.abs(audio).max() if len(audio) > 0 else 0
    print(f"[debug] Captured {duration_sec:.2f}s, max amplitude: {max_amplitude} (int16 range is 0-32767)")
    if max_amplitude < 500:
        print("[debug] WARNING: amplitude very low — check mic volume/selection")


# ---------------------------------------------------------------------------
# Step 2: Transcribe with Faster-Whisper
# ---------------------------------------------------------------------------
def transcribe_audio(filename: str = AUDIO_IN_FILE) -> str:
    segments, _info = whisper_model.transcribe(filename)
    text = " ".join(segment.text for segment in segments)
    return text.strip()


def has_audio(filename: str = AUDIO_IN_FILE) -> bool:
    """True if the recording actually captured any audio (VAD may end up
    with an empty/near-empty file if the user never spoke)."""
    _rate, audio = read(filename)
    return len(audio) > FRAME_SAMPLES * MIN_SPEECH_FRAMES


# ---------------------------------------------------------------------------
# Step 3: Tools the agent can call
# ---------------------------------------------------------------------------
def get_current_time() -> str:
    return datetime.now().strftime("%I:%M %p")


def calculate(expression: str) -> str:
    # NOTE: eval() is unsafe for untrusted input in a real product.
    # Fine for this local learning project; we'll replace it with a
    # proper sandboxed evaluator when we harden this later.
    try:
        return str(eval(expression))  # noqa: S307
    except Exception:
        return "Unable to calculate the expression."


TOOLS_DESCRIPTION = """
Available tools:

1. get_current_time
   Use this when the user asks for the current time.

2. calculate
   Use this for mathematical calculations.

If a tool is required, respond ONLY with JSON in this exact format:
{"tool": "tool_name", "argument": "tool_argument"}

If no tool is required, just respond normally in plain text.
"""


# ---------------------------------------------------------------------------
# Step 4: Ask the local LLM
# ---------------------------------------------------------------------------
def ask_llm(user_text: str) -> str:
    prompt = f"""You are a helpful voice AI agent.

{TOOLS_DESCRIPTION}

User request:
{user_text}
"""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Step 5: Run a tool if the LLM asked for one
# ---------------------------------------------------------------------------
def process_response(response: str) -> str:
    try:
        tool_call = json.loads(response)
        tool_name = tool_call.get("tool")
        argument = tool_call.get("argument")

        if tool_name == "get_current_time":
            result = get_current_time()
        elif tool_name == "calculate":
            result = calculate(argument)
        else:
            return response

        return f"The result is {result}"
    except (json.JSONDecodeError, TypeError):
        return response


# ---------------------------------------------------------------------------
# Step 6: Speak the response with Piper
# ---------------------------------------------------------------------------
def speak(text: str) -> None:
    command = ["piper", "--model", PIPER_MODEL, "--output_file", AUDIO_OUT_FILE]

    t_tts_start = time.perf_counter()
    process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True)
    process.communicate(text)
    t_tts_generated = time.perf_counter()

    rate, audio = read(AUDIO_OUT_FILE)
    sd.play(audio, rate)
    sd.wait()
    t_playback_done = time.perf_counter()

    tts_generation_ms = (t_tts_generated - t_tts_start) * 1000
    playback_ms = (t_playback_done - t_tts_generated) * 1000
    print(f"[latency] TTS generation: {tts_generation_ms:.0f}ms | Playback: {playback_ms:.0f}ms")


# ---------------------------------------------------------------------------
# Step 7: The main loop
# ---------------------------------------------------------------------------
def run_voice_agent() -> None:
    print("Voice agent ready. Say 'exit', 'quit', or 'stop' to end.\n")
    while True:
        record_audio()

        if not has_audio():
            print("(Didn't hear anything, try again)")
            continue

        # --- Stage 2: STT ---
        t_stt_start = time.perf_counter()
        user_text = transcribe_audio()
        t_stt_end = time.perf_counter()
        print("You:", user_text)

        if user_text.lower().strip() in ("exit", "quit", "stop"):
            print("Voice agent stopped.")
            break

        if not user_text.strip():
            print("(Didn't catch anything, try again)")
            continue

        # --- Stage 3: LLM ---
        t_llm_start = time.perf_counter()
        llm_response = ask_llm(user_text)
        t_llm_end = time.perf_counter()

        final_response = process_response(llm_response)
        print("Agent:", final_response)

        # --- Stage 4: TTS (timed inside speak()) ---
        t_response_ready = time.perf_counter()
        speak(final_response)

        # --- Full breakdown: this is the latency that matters, i.e. the
        # time from "user stopped talking" to "agent starts responding" ---
        stt_ms = (t_stt_end - t_stt_start) * 1000
        llm_ms = (t_llm_end - t_llm_start) * 1000
        response_to_speak_ms = (t_response_ready - t_stt_start) * 1000
        print(
            f"[latency] STT: {stt_ms:.0f}ms | LLM: {llm_ms:.0f}ms | "
            f"Total (end-of-speech -> reply ready): {response_to_speak_ms:.0f}ms\n"
        )


if __name__ == "__main__":
    run_voice_agent()