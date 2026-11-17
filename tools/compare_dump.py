"""Audio-level distance between two TVQ1 dumps holding `audio` and `prosody` (e.g. C engine vs golden).
    python tools/compare_dump.py out/test0.bin out/c_test0.bin"""
import struct, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import quant_eval as qe          # noqa: E402

def load(path):
    data = open(path, "rb").read()
    _, _, n, _ = struct.unpack_from("<4sIII", data)
    out = {}
    for i in range(n):
        name, dtype, ndim, *rest = struct.unpack_from("<64sII4III", data, 16 + 96 * i)
        shape, off, nbytes = rest[:ndim], rest[4], rest[5]
        dt = {0: np.float32, 1: np.int8, 2: np.int32}[dtype]
        out[name.rstrip(b"\0").decode()] = np.frombuffer(data, dt, nbytes // np.dtype(dt).itemsize, off).reshape(shape)
    return out

a, b = load(sys.argv[1]), load(sys.argv[2])
ra = {"audio": torch.tensor(a["audio"].reshape(-1).copy()), "prosody": torch.tensor(a["prosody"].reshape(-1, 3).copy())}
rb = {"audio": torch.tensor(b["audio"].reshape(-1).copy()), "prosody": torch.tensor(b["prosody"].reshape(-1, 3).copy())}
mel, cents, flips = qe.compare(ra, rb)
print(f"mel L1 {mel:.2f} dB, F0 {cents:.1f} cents, voicing flips {flips:.2%}")
