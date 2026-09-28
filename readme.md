# Voice Agent — Base Version (100% local, no vendor SDKs)

A complete voice-in, voice-out AI agent running entirely on your machine:
- **STT**: Faster-Whisper (local)
- **LLM**: Llama 3.2 via Ollama (local)
- **TTS**: Piper (local)
- **Tools**: get_current_time, calculate

No API keys. No cloud calls. No cost.

---

## 1. Install Ollama and pull the model

Download Ollama from https://ollama.com (macOS/Windows/Linux installers available), then:

```bash
ollama pull llama3.2
```

Leave Ollama running in the background — it starts a local server automatically after install (usually on `localhost:11434`).

## 2. Install Python dependencies

```bash
pip install faster-whisper sounddevice scipy ollama numpy webrtcvad-wheels
```

## 3. Install Piper (TTS)

```bash
pip install piper-tts
```

Piper also needs a voice model file (`.onnx` + `.onnx.json`). Download one from:
https://github.com/rhasspy/piper/blob/master/VOICES.md

For this project we use `en_US-lessac-medium`. Download both:
- `en_US-lessac-medium.onnx`
- `en_US-lessac-medium.onnx.json`

Place both files in this same folder as `main.py`.

**Linux users** may also need `espeak-ng` installed system-wide:
```bash
sudo apt-get install espeak-ng
```

## 4. Run it

```bash
python main.py
```

Speak when you see "Listening..." — it waits for you to start talking, stops ~800 ms after
you go quiet, transcribes, thinks, and replies out loud. Say "exit", "quit", or "stop" to end.

## Troubleshooting

**`ModuleNotFoundError: No module named 'pkg_resources'` on `import webrtcvad`**
The original `webrtcvad` package (2.0.10) imports `pkg_resources`, which used to be
installed with `setuptools`. Python 3.12+ venvs no longer include `setuptools`, and
`setuptools>=81` has removed `pkg_resources` entirely — so this import now fails.
Fixed here by using **`webrtcvad-wheels`** instead: a maintained fork of the same
project (same `import webrtcvad`, same API) that ships prebuilt wheels for Python
3.12/3.13 and dropped the `pkg_resources` dependency. `main.py` needs no changes.

If you must stay on the original `webrtcvad`, install the last setuptools that still
ships `pkg_resources`:

```bash
pip install "setuptools<81"
```

**`ResponseError: llama-server process has terminated: exit status 0xc0000409 ... CUDA error: shared object initialization failed`**
Ollama's bundled CUDA build crashes while loading *any* model on this machine: the GPU
buffers are allocated fine (26 layers, ~2.2 GiB), then llama-server dies in llama.cpp's
PDL probe (`cudaFuncGetAttributes` in `ggml_cuda_kernel_can_use_pdl`) with
`CUDA error: shared object initialization failed`. No Python code can fix this — it is an
Ollama/llama.cpp + NVIDIA-driver issue (driver 592.00 / CUDA 13.1, Ollama 0.34.4).

Fixed here by running Ollama's **Vulkan** backend, which still uses the GPU:

```powershell
setx OLLAMA_VULKAN 1     # env var for the whole user account
```
Then quit Ollama from the system tray and start it again so the server picks it up.
Verify it worked: `ollama ps` should show the model loaded with the `PROCESSOR` column
*not* `100% CPU` (measured here: ~2.34 GB in VRAM, ~22 tokens/s vs ~4.5 tokens/s on CPU).

Alternatives if you prefer to stay on CUDA: update the NVIDIA driver and/or Ollama, then
remove the variable (`setx OLLAMA_VULKAN ""`). To force CPU-only instead (slower, always works):

```powershell
setx CUDA_VISIBLE_DEVICES -1
```
…or per request in `main.py`: `ollama.chat(..., options={"num_gpu": 0})`.

## Upgrade log

- ✅ **v2.2 — Ollama GPU fix (Vulkan backend):** Ollama's CUDA build died on model load
  (`0xc0000409` / `CUDA error: shared object initialization failed`) on this RTX 2050.
  Switched Ollama to `OLLAMA_VULKAN=1` (still GPU, ~22 tok/s vs ~4.5 on CPU).
  See Troubleshooting above. `main.py` unchanged.
- ✅ **v2.1 — Dependency fix (Python 3.13):** swapped `webrtcvad` → `webrtcvad-wheels`
  (same module/API) because the old package's hard `pkg_resources` import breaks on
  Python 3.12+ venvs and `setuptools>=81`. See Troubleshooting above.
- ✅ **v2 — Real endpointing (VAD):** replaced the fixed 5-second recording with
  `webrtcvad`-based silence detection. It now waits for you to start speaking,
  then stops automatically ~800ms after you go quiet, instead of always
  recording a flat 5 seconds. Tune this in `main.py`:
  - `VAD_AGGRESSIVENESS` (0–3): higher = more strictly filters out background
    noise as "not speech," but can also clip soft speech.
  - `SILENCE_MS_TO_STOP`: lower = snappier but more likely to cut you off
    mid-thought; higher = safer but feels laggier. This is the exact
    latency-vs-interruption trade-off we discussed — tune it by testing your
    own natural pauses.

## Known limitations (still on purpose — upcoming upgrades)

- **Fully batch, no streaming** — each stage still waits for the last to finish completely, so latency is the sum of all four stages
- **No interruption handling** — you can't talk over the agent while it's speaking
- **Manual JSON tool-calling** — brittle if the model doesn't format its response exactly right