"""Audio-level distance between two TVQ1 dumps holding `audio` and `prosody` (e.g. board vs host).

    python tools/compare_dump.py out/c.bin out/dev.bin
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tvq1                                                   # noqa: E402
from audio_metrics import compare                             # noqa: E402


def load(path):
    d = tvq1.read(path)
    return {"audio": torch.tensor(d["audio"].reshape(-1).copy()), "prosody": torch.tensor(d["prosody"].reshape(-1, 3).copy())}


mel, cents, flips = compare(load(sys.argv[1]), load(sys.argv[2]))
print(f"mel L1 {mel:.2f} dB, F0 {cents:.1f} cents, voicing flips {flips:.2%}")
