"""Realtime English speech-to-text with NVIDIA Parakeet TDT 0.6B v2.

Browser mic -> WebSocket (16 kHz mono PCM16) -> Silero VAD -> Parakeet (onnx-asr).

Why it is built this way: Parakeet is an offline model. Feeding it fixed-size
chunks raises its error rate a lot (published chunked results run ~9-17% WER vs
~6% on whole utterances). So this server waits for a pause, transcribes the whole
utterance in one call for the final text, and only uses quick re-transcriptions of
the utterance-so-far as a live preview while you are still talking.

Run:
    python parakeet_server.py                          # auto: GPU if available, else CPU
    python parakeet_server.py --device cpu --quantization int8   # smaller/faster on CPU
Then open http://localhost:8000
"""
import argparse
import asyncio
import collections
import json
import logging
import os
import pathlib
import time
import urllib.request

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

SR = 16000
VAD_FRAME = 512          # Silero VAD frame at 16 kHz (32 ms)
VAD_CONTEXT = 64         # samples of previous frame Silero v5 expects in front of each frame
VAD_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
HERE = pathlib.Path(__file__).resolve().parent

log = logging.getLogger("parakeet")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def pick_providers(device: str) -> list:
    if device in ("auto", "cuda"):
        # onnxruntime-gpu >= 1.21 can load the CUDA/cuDNN libraries that pip installed
        # (e.g. the ones torch brings on Colab) so they do not have to be on the path.
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception as e:  # noqa: BLE001
                log.warning("preload_dlls failed: %s", e)
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda":
            log.warning("CUDA requested but onnxruntime has no CUDAExecutionProvider "
                        "(is onnxruntime-gpu installed, and the CPU-only onnxruntime removed?). "
                        "Falling back to CPU.")
    return ["CPUExecutionProvider"]


def session_providers(obj, depth=0, seen=None) -> list:
    """Find the providers the loaded onnxruntime sessions actually ended up using."""
    seen = seen if seen is not None else set()
    if depth > 4 or id(obj) in seen:
        return []
    seen.add(id(obj))
    if isinstance(obj, ort.InferenceSession):
        return obj.get_providers()
    found = []
    for v in getattr(obj, "__dict__", {}).values():
        found += session_providers(v, depth + 1, seen)
        if found:
            break
    return found


def load_vad_session() -> ort.InferenceSession:
    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "parakeet_server"
    path = cache / "silero_vad.onnx"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        log.info("downloading Silero VAD -> %s", path)
        urllib.request.urlretrieve(VAD_URL, path)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


class Vad:
    """Streaming Silero VAD. One instance per connection (it carries recurrent state)."""

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.sr = np.array(SR, dtype=np.int64)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, VAD_CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.context, frame[None, :]], axis=1)
        out, self.state = self.session.run(None, {"input": x, "state": self.state, "sr": self.sr})
        self.context = x[:, -VAD_CONTEXT:]
        return float(out[0][0])


# ---------------------------------------------------------------------------
# Per-connection segmentation
# ---------------------------------------------------------------------------

class Segmenter:
    """Turns a stream of VAD frames into utterances and asks for previews/finals."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.preroll = collections.deque(maxlen=max(1, int(cfg.preroll * SR / VAD_FRAME)))
        self.utt: list[np.ndarray] = []
        self.in_utt = False
        self.silence_frames = 0
        self.utt_id = 0

    @property
    def utt_seconds(self) -> float:
        return len(self.utt) * VAD_FRAME / SR

    def push(self, frame: np.ndarray, prob: float):
        """Returns "final" when the current utterance should be finalized, else None."""
        c = self.cfg
        if not self.in_utt:
            if prob >= c.vad_start:
                self.in_utt = True
                self.utt_id += 1
                self.utt = list(self.preroll) + [frame]
                self.preroll.clear()
                self.silence_frames = 0
            else:
                self.preroll.append(frame)
            return None

        self.utt.append(frame)
        self.silence_frames = self.silence_frames + 1 if prob < c.vad_end else 0
        silence_s = self.silence_frames * VAD_FRAME / SR
        if silence_s >= c.end_silence:
            return "final"
        # Long utterance: accept a shorter pause so text is not held back forever.
        if self.utt_seconds >= c.soft_max and silence_s >= 0.25:
            return "final"
        if self.utt_seconds >= c.hard_max:
            return "final"
        return None

    def take(self) -> np.ndarray:
        audio = np.concatenate(self.utt) if self.utt else np.zeros(0, dtype=np.float32)
        self.utt = []
        self.in_utt = False
        self.silence_frames = 0
        return audio


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def build_app(cfg) -> FastAPI:
    import onnx_asr

    providers = pick_providers(cfg.device)
    log.info("loading %s (quantization=%s) with %s", cfg.model, cfg.quantization, providers)
    t = time.time()
    model = onnx_asr.load_model(cfg.model, quantization=cfg.quantization, providers=providers)
    load_s = time.time() - t
    used = session_providers(model) or providers
    vad_session = load_vad_session()

    # Warm up so the first real request is not slow (CUDA kernels, memory arenas).
    warm = (np.random.default_rng(0).standard_normal(SR * 2) * 0.01).astype(np.float32)
    t = time.time()
    model.recognize(warm, sample_rate=SR)
    warm_ms = (time.time() - t) * 1000
    device = "GPU (CUDA)" if used and used[0] == "CUDAExecutionProvider" else "CPU"
    log.info("ready: %s on %s | load %.1fs | warmup %.0f ms", cfg.model, device, load_s, warm_ms)

    infer_lock = asyncio.Lock()   # one inference at a time on the shared model

    async def recognize(audio: np.ndarray) -> tuple[str, float]:
        async with infer_lock:
            t0 = time.time()
            text = await asyncio.to_thread(model.recognize, audio, sample_rate=SR)
            return (text or "").strip(), (time.time() - t0) * 1000

    app = FastAPI()
    info = {"model": cfg.model, "device": device, "providers": used, "quantization": cfg.quantization}

    @app.get("/")
    async def index():
        return HTMLResponse((HERE / "parakeet_ui.html").read_text(encoding="utf-8"))

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok", **info, "warmup_ms": round(warm_ms)})

    @app.websocket("/asr")
    async def asr(ws: WebSocket):
        await ws.accept()
        await ws.send_json({"type": "ready", **info})
        log.info("client connected")

        seg = Segmenter(cfg)
        vad = Vad(vad_session)
        jobs: asyncio.Queue = asyncio.Queue()
        state = {"preview_busy": False, "last_preview": 0.0}
        pending = np.zeros(0, dtype=np.float32)
        leftover = b""

        async def worker():
            while True:
                kind, uid, audio = await jobs.get()
                try:
                    if kind == "flushed":
                        await ws.send_json({"type": "flushed"})
                        continue
                    if kind == "preview" and not jobs.empty():
                        # A final for this utterance is already queued; the preview is stale.
                        state["preview_busy"] = False
                        continue
                    if audio is None:   # too short to be speech: just clear the preview
                        await ws.send_json({"type": "final", "id": uid, "text": "", "ms": 0, "audio_s": 0})
                        continue
                    try:
                        text, ms = await recognize(audio)
                    except Exception:  # noqa: BLE001  (onnxruntime errors subclass RuntimeError)
                        state["preview_busy"] = False
                        log.exception("inference failed for %s #%d", kind, uid)
                        continue
                    dur = len(audio) / SR
                    if kind == "final":
                        if text:
                            log.info("[final #%d] %.1fs audio, %.0f ms: %s", uid, dur, ms, text)
                        await ws.send_json({"type": "final", "id": uid, "text": text,
                                            "ms": round(ms), "audio_s": round(dur, 2)})
                    else:
                        state["preview_busy"] = False
                        if seg.in_utt and seg.utt_id == uid:   # still the same utterance
                            await ws.send_json({"type": "partial", "id": uid, "text": text, "ms": round(ms)})
                except (WebSocketDisconnect, RuntimeError):
                    return   # socket closed while we were sending

        def finalize():
            uid = seg.utt_id
            audio = seg.take()
            enough = len(audio) >= int(cfg.min_utterance * SR)
            jobs.put_nowait(("final", uid, audio if enough else None))

        wtask = asyncio.create_task(worker())
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("text") is not None:
                    if msg["text"] == "flush":
                        if seg.in_utt:
                            finalize()
                        jobs.put_nowait(("flushed", 0, None))
                    continue
                data = leftover + (msg.get("bytes") or b"")
                cut = len(data) - (len(data) % 2)
                leftover = data[cut:]
                if not cut:
                    continue
                pcm = np.frombuffer(data[:cut], dtype="<i2").astype(np.float32) / 32768.0
                pending = np.concatenate([pending, pcm])

                n = len(pending) // VAD_FRAME
                for k in range(n):
                    frame = pending[k * VAD_FRAME:(k + 1) * VAD_FRAME]
                    if seg.push(frame, vad(frame)) == "final":
                        finalize()
                pending = pending[n * VAD_FRAME:]

                # Live preview of the utterance so far (never more than one in flight).
                now = time.time()
                if (seg.in_utt and not state["preview_busy"]
                        and seg.utt_seconds >= 0.5
                        and now - state["last_preview"] >= cfg.preview_every):
                    state["preview_busy"] = True
                    state["last_preview"] = now
                    jobs.put_nowait(("preview", seg.utt_id, np.concatenate(seg.utt)))
        except WebSocketDisconnect:
            pass
        finally:
            wtask.cancel()
            log.info("client disconnected")

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nemo-parakeet-tdt-0.6b-v2", help="onnx-asr model name (English: nemo-parakeet-tdt-0.6b-v2)")
    ap.add_argument("--quantization", default=None, help="None (full precision) or int8 (smaller, faster on CPU)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--end-silence", type=float, default=0.6, help="pause (s) that ends an utterance")
    ap.add_argument("--soft-max", type=float, default=12.0, help="after this many seconds, a 0.25s pause also ends an utterance")
    ap.add_argument("--hard-max", type=float, default=20.0, help="force-finalize utterances longer than this (s)")
    ap.add_argument("--preview-every", type=float, default=0.5, help="min seconds between live previews")
    ap.add_argument("--preroll", type=float, default=0.3, help="audio (s) kept from before speech starts")
    ap.add_argument("--vad-start", type=float, default=0.5, help="speech probability that starts an utterance")
    ap.add_argument("--vad-end", type=float, default=0.35, help="speech probability below which a frame counts as silence")
    ap.add_argument("--min-utterance", type=float, default=0.3, help="ignore utterances shorter than this (s)")
    cfg = ap.parse_args()
    if cfg.quantization in ("", "none", "None"):
        cfg.quantization = None

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = build_app(cfg)
    print(f"Open http://localhost:{cfg.port}", flush=True)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
