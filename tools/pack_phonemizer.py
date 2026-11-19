"""Pack dizione's data files into one TVQ1 container (raw int8 blobs) for the `phonemizer`
flash partition of the ESP32-S3 firmware.

    python tools/pack_phonemizer.py $TINY_VOCA_WEIGHTS/phonemizer out/phonemizer.bin
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tvq1                                                   # noqa: E402

FILES = ("lexicon.tsv", "align.json", "g2p.json", "g2p.npz", "homographs.json")
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
blobs = [(name, (src / name).read_bytes()) for name in FILES]
size = tvq1.write_raw(dst, [(name, tvq1.I8, (len(data),), data) for name, data in blobs])
print(f"{dst}: {size / 1e6:.2f} MB, {', '.join(n for n, _ in blobs)}")
