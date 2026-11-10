"""Patch WhisperLiveKit's browser UI so the mic actually streams audio.

Problem (WhisperLiveKit 0.2.26, --pcm-input / AudioWorklet path):
  * The AudioContext is created after an `await`, so Chrome's autoplay policy can
    start it suspended; nothing ever calls resume(). A suspended context renders
    nothing.
  * Nothing in the audio graph is connected to audioContext.destination, and the
    PCM worklet is built with numberOfOutputs: 0 so it cannot be connected either.
    An unreachable graph is not pulled by the renderer.
    Either one stalls capture: the waveform stays flat, the server sees ~0s of audio.
  * Even once audio flows, the worklet posts ONE render quantum (128 frames) per
    WebSocket message - ~86-byte frames at ~375 msg/s. The server cannot keep up.
    Measured on identical audio: 43-sample chunks transcribe as "You", while
    1600-sample chunks give "jumps over the lazy dog."

Fix: resume the context, give the worklet an output, route the analyser and the
worklet into a muted (gain = 0) sink on the destination so the graph is rendered,
and buffer ~100 ms of audio per message. Gain is 0, so nothing is played back and
there is no feedback loop.

Run after every `pip install`/upgrade of whisperlivekit (venv/ is gitignored):
    venv\\Scripts\\python.exe patch_webui.py
Then just refresh http://localhost:8000 - the server rebuilds the page from disk
on each request, so no restart is needed.
"""
import sys
from pathlib import Path

MARKER = "/* wlk-mic-fix */"

ANALYSER_OLD = """    microphone = audioContext.createMediaStreamSource(stream);
    microphone.connect(analyser);
"""

ANALYSER_NEW = """    microphone = audioContext.createMediaStreamSource(stream);
    microphone.connect(analyser);
    %s
    // Chrome starts an AudioContext suspended when it is created after an await,
    // and it does not render a graph that never reaches the destination. Resume it
    // and add a muted sink so the analyser and the PCM worklet are actually pulled.
    console.log("[wlk-mic-fix] AudioContext state before resume=" + audioContext.state);
    if (audioContext.state === "suspended") {
      try { await audioContext.resume(); } catch (e) { console.warn("[wlk-mic-fix] resume failed", e); }
    }
    keepAliveSink = audioContext.createGain();
    keepAliveSink.gain.value = 0;
    keepAliveSink.connect(audioContext.destination);
    analyser.connect(keepAliveSink);
    console.log("[wlk-mic-fix] AudioContext state=" + audioContext.state +
                " sampleRate=" + audioContext.sampleRate);
""" % MARKER

WORKLET_OLD = """      workletNode = new AudioWorkletNode(audioContext, "pcm-forwarder", { numberOfInputs: 1, numberOfOutputs: 0, channelCount: 1 });
      microphone.connect(workletNode);
"""

WORKLET_NEW = """      workletNode = new AudioWorkletNode(audioContext, "pcm-forwarder", { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], channelCount: 1 });
      microphone.connect(workletNode);
      %s
      // give the worklet a (silent) path to the destination so process() keeps running
      workletNode.connect(keepAliveSink);
      workletChunkCount = 0;
""" % MARKER

# Declare the two new globals next to the existing audio ones.
GLOBALS_OLD = "let outputAudioContext = null;\n"
GLOBALS_NEW = "let outputAudioContext = null;\nlet keepAliveSink = null; %s\nlet workletChunkCount = 0;\n" % MARKER

# Count forwarded chunks so the console shows whether capture is alive.
COUNT_OLD = """      workletNode.port.onmessage = (e) => {
        const data = e.data;
"""

COUNT_NEW = """      workletNode.port.onmessage = (e) => {
        const data = e.data;
        if (++workletChunkCount %% 100 === 1) {
          console.log("[wlk-mic-fix] worklet chunks forwarded: " + workletChunkCount);
        }
""" % ()


# The stock worklet posts every 128-frame render quantum straight down the
# WebSocket: ~86-byte frames at ~375 msg/s. The server cannot keep up and the
# transcript degrades to nothing. Buffer ~100 ms per message instead.
# NOTE: web_interface.py inlines this file inside a JS template literal, so this
# code must contain no backticks and no "${".
WORKLET_JS = """%s
// Buffers ~100 ms of audio per message. Sending one render quantum (128 frames)
// per WebSocket frame overwhelms the server: same audio, 43-sample chunks
// transcribe as "You" where 1600-sample chunks give the full sentence.
var WLK_FRAMES_PER_MESSAGE = 4800;

class PCMForwarder extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(WLK_FRAMES_PER_MESSAGE);
    this._n = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input[0] && input[0].length) {
      const channelData = input[0];
      for (let i = 0; i < channelData.length; i++) {
        this._buf[this._n++] = channelData[i];
        if (this._n === this._buf.length) {
          const copy = new Float32Array(this._buf);
          this.port.postMessage(copy, [copy.buffer]);
          this._n = 0;
        }
      }
    }
    // Keep processor alive
    return true;
  }
}

registerProcessor('pcm-forwarder', PCMForwarder);
""" % MARKER


def write_worklet(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"  {path.name}: already patched, skipping")
        return False
    if "`" in WORKLET_JS or "${" in WORKLET_JS:
        sys.exit("ERROR: worklet code must not contain backticks or ${ (it is inlined into a template literal)")
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"  {path.name}: backed up to {backup.name}")
    path.write_text(WORKLET_JS, encoding="utf-8")
    print(f"  {path.name}: rewritten with ~100ms buffering")
    return True


def patch(path: Path, pairs) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"  {path.name}: already patched, skipping")
        return False
    for old, new in pairs:
        if old not in text:
            sys.exit(f"ERROR: expected snippet not found in {path.name}:\n{old}")
        text = text.replace(old, new, 1)
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  {path.name}: backed up to {backup.name}")
    path.write_text(text, encoding="utf-8")
    print(f"  {path.name}: patched")
    return True


def main():
    import whisperlivekit

    web = Path(whisperlivekit.__file__).parent / "web"
    js = web / "live_transcription.js"
    if not js.exists():
        sys.exit(f"ERROR: {js} not found")
    print(f"Patching {web}")
    patch(js, [
        (GLOBALS_OLD, GLOBALS_NEW),
        (ANALYSER_OLD, ANALYSER_NEW),
        (WORKLET_OLD, WORKLET_NEW),
        (COUNT_OLD, COUNT_NEW),
    ])
    write_worklet(web / "pcm_worklet.js")
    print("Done. Refresh http://localhost:8000 (hard-refresh: Ctrl+Shift+R).")


if __name__ == "__main__":
    main()
