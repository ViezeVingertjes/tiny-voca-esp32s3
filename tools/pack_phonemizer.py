"""Pack dizione's data files into one TVQ1 container (raw int8 blobs) for the `phonemizer`
flash partition of the ESP32-S3 firmware.

    python esp32/tools/pack_phonemizer.py ../models/tiny-voca/phonemizer esp32/out/phonemizer.bin
"""
import struct, sys
from pathlib import Path

FILES = ("lexicon.tsv", "align.json", "g2p.json", "g2p.npz", "homographs.json")
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
blobs = [(name, (src / name).read_bytes()) for name in FILES]
offset = (16 + 96 * len(blobs) + 15) & ~15
entries, placed = [], []
for name, data in blobs:
    entries.append(struct.pack("<64sII4III", name.encode(), 1, 1, len(data), 1, 1, 1, offset, len(data)))
    placed.append((offset, data))
    offset = (offset + len(data) + 15) & ~15
with open(dst, "wb") as f:
    f.write(b"TVQ1" + struct.pack("<III", 1, len(blobs), 0) + b"".join(entries))
    for off, data in placed:
        f.write(b"\0" * (off - f.tell()))
        f.write(data)
print(f"{dst}: {offset / 1e6:.2f} MB, {', '.join(n for n, _ in blobs)}")
