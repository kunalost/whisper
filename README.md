# whisper

Local, multilingual, real-time speech-to-text using Whisper (`faster-whisper`) served through
[WhisperLiveKit](https://pypi.org/project/whisperlivekit/). Speak into the browser mic and see text live.

## Setup (Windows)

Use Python 3.12 - `ctranslate2` has no wheels for 3.14 yet.

```powershell
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install faster-whisper sounddevice numpy whisperlivekit
venv\Scripts\python.exe patch_webui.py
```

On a machine with an NVIDIA GPU, also install the CUDA runtime libs:

```powershell
venv\Scripts\python.exe -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

`patch_webui.py` is required - without it the browser connects but no audio ever
reaches the server ("No audio detected..."). See **Browser mic fix** below. It edits
files inside `venv/`, so re-run it after any `pip install`/upgrade of whisperlivekit.

## Run the live server

```powershell
.\start_server.ps1                          # auto-detects GPU/CPU, picks a sensible model
.\start_server.ps1 -Language hi             # force Hindi
.\start_server.ps1 -Model large-v3-turbo    # faster than large-v3, slightly less accurate
.\start_server.ps1 -Model tiny              # lowest lag on CPU
.\start_server.ps1 -Device cpu -CpuThreads 8
```

Then open http://localhost:8000 and allow the microphone.

Defaults are chosen per device: `large-v3` + `int8_float16` on GPU, `base` + `int8` on CPU.

## Files

- `start_server.ps1` - starts the server (auto-detects CUDA, device/model/language options).
- `run_wlk.py` - launcher that forces faster-whisper device/compute type (WhisperLiveKit hardcodes `auto`).
- `patch_webui.py` - fixes the browser mic capture path (required, see below).
- `realtime.py` - standalone terminal mic transcription script (no browser).

## Browser mic fix

WhisperLiveKit 0.2.26's `--pcm-input` browser path has two bugs that together mean
the mic produces nothing. `patch_webui.py` fixes both and keeps `.orig` backups.

1. **The audio graph is never rendered.** The PCM worklet is created with
   `numberOfOutputs: 0` and nothing in the graph connects to `audioContext.destination`,
   so Chrome does not pull it. Symptom: flat waveform, server receives ~0.1s of audio
   no matter how long you talk. Fix: give the worklet an output and route it plus the
   analyser through a muted (`gain = 0`) sink on the destination.
2. **Chunks are far too small.** The worklet posted one render quantum (128 frames)
   per WebSocket message - ~86-byte frames at ~375 msg/s - and the server cannot keep
   up. Measured on identical audio: 43-sample chunks transcribe as `"You"`, while
   1600-sample chunks give `"jumps over the lazy dog."` Fix: buffer ~100 ms per message.

## Notes

- `--pcm-input` is used so FFmpeg is not required for live streaming. The REST endpoint
  `POST /v1/audio/transcriptions` (file upload) *does* shell out to FFmpeg and returns 500
  without it; the live WebSocket path is unaffected.
- `--backend-policy localagreement` is used because the default `simulstreaming` policy hit a
  recursion bug with faster-whisper's encoder output in WhisperLiveKit 0.2.26.
- `--min-chunk-size` defaults to 0.1s upstream, i.e. ~10 inference attempts per second.
  On CPU a single pass takes about a second, so work piles up and the transcript falls
  minutes behind. `start_server.ps1` raises it to 1.0s on CPU.
- **CPU throughput** (12-thread CPU, int8, 30s of English speech, batch):

  | model | speed | note |
  |-------|-------|------|
  | tiny  | 2.31x realtime | lowest lag |
  | base  | 1.55x realtime | default on CPU |
  | small | 0.52x realtime | slower than realtime, never catches up |

  Streaming costs more than batch, so even `base` accumulates lag during long
  stretches of unbroken speech; natural pauses let it catch up. `large-v3` is
  GPU-only in practice.
- Whisper accuracy on Hindi/Hinglish is well below English, and the smaller CPU models are
  weaker still; auto language detection can pick the wrong language in live streaming.
  Use `-Language hi` for Hindi speech.
- A single model runs on either GPU or CPU, not both at once.
