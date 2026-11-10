"""Realtime multilingual mic transcription (faster-whisper, large-v3-turbo, GPU).

Run:  venv\\Scripts\\python.exe realtime.py            (auto language detect)
      venv\\Scripts\\python.exe realtime.py --lang hi   (force Hindi)
Stop with Ctrl+C.
"""
import argparse
import os
import queue
import sys
import time

# --- make pip-installed CUDA libs (cublas/cudnn) visible to ctranslate2 on Windows ---
import site
for sp in site.getsitepackages():
    nv = os.path.join(sp, "nvidia")
    if os.path.isdir(nv):
        for sub in os.listdir(nv):
            b = os.path.join(nv, sub, "bin")
            if os.path.isdir(b):
                os.add_dll_directory(b)
                os.environ["PATH"] = b + os.pathsep + os.environ["PATH"]

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SR = 16000
BLOCK = 1600            # 100 ms
SILENCE_RMS = 0.012     # energy threshold for "speech"
END_SILENCE_S = 0.8     # silence that ends an utterance
PARTIAL_EVERY_S = 1.0   # how often to refresh live text
MAX_UTT_S = 25          # force-finalize long utterances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="default: large-v3-turbo on GPU, base on CPU")
    ap.add_argument("--lang", default=None, help="e.g. hi, en, es. Default: auto-detect")
    ap.add_argument("--compute", default=None, help="float16 | int8_float16 | int8")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    ap.add_argument("--mic", type=int, default=None, help="input device index")
    ap.add_argument("--list-mics", action="store_true")
    a = ap.parse_args()

    if a.list_mics:
        print(sd.query_devices())
        return

    # Resolve device/model/compute. large-v3-turbo on CPU is far slower than
    # realtime, so fall back to a model that can actually keep up.
    if a.device == "auto":
        import ctranslate2
        a.device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        print(f"Auto-detected device: {a.device}")
    on_cpu = a.device == "cpu"
    if a.model is None:
        a.model = "base" if on_cpu else "large-v3-turbo"
    if a.compute is None:
        a.compute = "int8" if on_cpu else "float16"
    if on_cpu and "float16" in a.compute:
        print("float16 is not supported on CPU, using int8.")
        a.compute = "int8"

    print(f"Loading {a.model} on {a.device} ({a.compute})... (first run downloads the model)")
    model = WhisperModel(a.model, device=a.device, compute_type=a.compute)
    print("Ready. Bolo... (Ctrl+C to stop)\n")

    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    def transcribe(audio):
        segs, info = model.transcribe(
            audio, language=a.lang, task=a.task, beam_size=1,
            vad_filter=True, condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segs).strip()
        return text, info.language, info.language_probability

    buf = np.zeros(0, dtype=np.float32)
    speaking = False
    last_voice = 0.0
    last_partial = 0.0

    def show(text, lang, prob, final):
        line = f"[{lang} {prob:.0%}] {text}"
        sys.stdout.write("\r\033[K" + line + ("\n" if final else ""))
        sys.stdout.flush()

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=BLOCK, device=a.mic, callback=cb):
        try:
            while True:
                chunk = q.get()
                now = time.time()
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                if rms > SILENCE_RMS:
                    speaking = True
                    last_voice = now
                if speaking:
                    buf = np.concatenate([buf, chunk])
                    dur = len(buf) / SR
                    silent_for = now - last_voice
                    if silent_for >= END_SILENCE_S or dur >= MAX_UTT_S:
                        text, lang, prob = transcribe(buf)
                        if text:
                            show(text, lang, prob, final=True)
                        buf = np.zeros(0, dtype=np.float32)
                        speaking = False
                    elif now - last_partial >= PARTIAL_EVERY_S and dur > 0.6:
                        text, lang, prob = transcribe(buf)
                        if text:
                            show(text, lang, prob, final=False)
                        last_partial = now
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
