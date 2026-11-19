"""Text -> phonemes on the host -> synthesis on the ESP32-S3 -> WAV, with a keep-up report.

    python tools/speak_s3.py --text "Hello there. How are you today?" --out out/s3.wav

The firmware (firmware/) must be running its command server. Phonemes come from the C++
dizione port (out/bin/dizione_cli, see tools/dizione_cli.cpp for the build); without that
binary a plain lexicon lookup is used and words missing from the lexicon are skipped.
Keep-up model: sentences are synthesized one after another and each plays as soon as it is
ready and the previous one has finished; any wait between sentences is a gap.
"""
import argparse, base64, signal, subprocess, sys, time, wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_voca_src import REPO, WEIGHTS, load_lexicon, phonemize           # noqa: E402  (also puts tiny_voca on the path)
from tiny_voca.infer import sentences, finished                          # noqa: E402
import tvserial                                                           # noqa: E402


class Dizione:
    """The C++ phonemizer as a subprocess: one sentence per line in, ids and stress out."""
    def __init__(self, binary, data):
        self.p = subprocess.Popen([binary, data], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def __call__(self, sentence):
        self.p.stdin.write(sentence.replace("\n", " ") + "\n")
        self.p.stdin.flush()
        out = self.p.stdout.readline().rstrip("\n")
        if out.startswith("ERR"):
            raise ValueError(out)
        ids, stress = out.split("\t")
        return [int(v) for v in ids.split(",")], [int(v) for v in stress.split(",")]


class Board:
    def __init__(self, port, profile=False):
        self.s = tvserial.open_port(port)
        self.buf = b""
        self.profile = profile

    def line(self, deadline):
        while b"\n" not in self.buf:
            if time.time() > deadline:
                raise TimeoutError("no answer from the board")
            self.buf += self.s.read(65536)
        raw, self.buf = self.buf.split(b"\n", 1)
        return raw.decode(errors="replace").rstrip("\r")

    def wait_ready(self, seconds):
        deadline = time.time() + seconds
        self.s.write(b"\n")
        while True:
            line = self.line(deadline)
            if line == "READY":
                return
            if any(m in line for m in tvserial.PANIC):
                raise RuntimeError(line)

    def synthesize(self, ids, stress, voice, seed, seconds):
        cmd = f"SYN {voice} {seed} {len(ids)} {','.join(map(str, ids))} {','.join(map(str, stress))}\n"
        self.s.write(cmd.encode())
        deadline = time.time() + seconds
        result, pcm, chunks_, in_dump = None, None, [], False
        while True:
            line = self.line(deadline)
            if line.startswith("RESULT "):
                result = dict(kv.split("=") for kv in line.split()[1:])
            elif line.startswith("TVDUMP pcm"):
                in_dump, chunks_ = True, []
            elif line.startswith("TVDUMP_END"):
                pcm, in_dump = base64.b64decode("".join(chunks_)), False
            elif in_dump:
                chunks_.append(line)
            elif line.startswith("ERR") or any(m in line for m in tvserial.PANIC):
                raise RuntimeError(line)
            elif line == "DONE":
                return result, pcm
            elif line.startswith("  ") and self.profile:
                print("   ", line.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", choices=("F2", "M2"), default="M2")
    parser.add_argument("--port", default=tvserial.default_port())
    parser.add_argument("--out", default=str(REPO / "out/s3.wav"))
    parser.add_argument("--phonemizer", default=str(WEIGHTS / "phonemizer"))
    parser.add_argument("--dizione", default=str(REPO / "out/bin/dizione_cli"))
    parser.add_argument("--timeout", type=int, default=600, help="hard limit for the whole run, seconds")
    parser.add_argument("--profile", action="store_true", help="print the board's per-op profile")
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, lambda *_: sys.exit("[speak_s3: hard timeout]"))
    signal.alarm(args.timeout)

    if Path(args.dizione).exists():
        dizione = Dizione(args.dizione, args.phonemizer)
        to_tokens = lambda text: (dizione(text), [])
    else:
        print("(dizione_cli not built: falling back to a plain lexicon lookup)")
        lexicon = load_lexicon(str(Path(args.phonemizer) / "lexicon.tsv"))
        def to_tokens(text):
            missing = []
            p = phonemize(text, lexicon, missing)
            return (list(p.ids), list(p.stress)), missing
    voice = 0 if args.voice == "F2" else 1
    board = Board(args.port, args.profile)
    board.wait_ready(30)
    pcm_all, rows, gap = [], [], int(0.25 * 24000)
    print(f"{'#':>2} {'tokens':>6} {'audio s':>7} {'synth s':>7} {'RTF':>5}  text")
    for i, sentence in enumerate(sentences(finished(args.text))):
        (ids, stress), missing = to_tokens(sentence)
        if missing:
            print(f"   (skipped words not in the lexicon: {', '.join(missing)})")
        result, pcm = board.synthesize(ids, stress, voice, seed=i, seconds=300)
        board.wait_ready(10)
        audio_s, synth_s = float(result["audio_ms"]) / 1000, float(result["synth_ms"]) / 1000
        rows.append((audio_s, synth_s))
        print(f"{i:2d} {len(ids):6d} {audio_s:7.2f} {synth_s:7.2f} {audio_s / synth_s:5.2f}  {sentence}"
              + (f"  [saturations {result['saturations']}]" if result["saturations"] != "0" else ""))
        pcm_all.append(pcm + b"\0\0" * gap)
    with wave.open(args.out, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(24000); f.writeframes(b"".join(pcm_all))
    # keep-up: sequential synthesis, playback starts as soon as a sentence is ready
    ready, play_end, gaps = 0.0, None, 0.0
    for audio_s, synth_s in rows:
        ready += synth_s
        start = ready if play_end is None else max(ready, play_end)
        if play_end is not None:
            gaps += start - play_end
        play_end = start + audio_s
    total_audio, total_synth = sum(a for a, _ in rows), sum(s for _, s in rows)
    print(f"\ntotal: {total_audio:.2f} s of audio in {total_synth:.2f} s (RTF {total_audio / total_synth:.2f}); "
          f"first audio after {rows[0][1]:.2f} s; gaps while playing: {gaps:.2f} s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
