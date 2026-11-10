"""Launcher: runs WhisperLiveKit server but forces faster-whisper device/compute settings.

WhisperLiveKit hardcodes device="auto", compute_type="auto". We patch
faster_whisper.WhisperModel so env vars can override them:
  WLK_DEVICE        cuda | cpu | auto      (default: cuda)
  WLK_COMPUTE       int8_float16 | float16 | int8 | auto   (default: int8_float16)
  WLK_CPU_THREADS   int, only matters when running on CPU
"""
import os
import sys

import faster_whisper

_orig_init = faster_whisper.WhisperModel.__init__


def _patched_init(self, *args, **kwargs):
    kwargs["device"] = os.environ.get("WLK_DEVICE", "cuda")
    kwargs["compute_type"] = os.environ.get("WLK_COMPUTE", "int8_float16")
    threads = int(os.environ.get("WLK_CPU_THREADS", "0"))
    if threads:
        kwargs["cpu_threads"] = threads
    print(f"[run_wlk] WhisperModel device={kwargs['device']} compute={kwargs['compute_type']} "
          f"cpu_threads={kwargs.get('cpu_threads', 'default')}", flush=True)
    _orig_init(self, *args, **kwargs)


faster_whisper.WhisperModel.__init__ = _patched_init

from whisperlivekit.basic_server import main  # noqa: E402

sys.argv = ["whisperlivekit-server"] + sys.argv[1:]
main()
