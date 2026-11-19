"""Export tiny-voca for the ESP32-S3 engine, plus a test vector from the golden model.

    python tools/export.py            # writes out/tvoca.bin, out/test0.bin and out/test0_dev.bin
    (test<N>.bin with --test-sentence N; test<N>_dev.bin holds only what the board needs)

All three are TVQ1 containers (tools/tvq1.py).

Per int8 linear `<base>` (convolutions as [out, 3*in] matrices on [x[t-1], x[t], x[t+1]]):
    <base>.wq   i8  [Npad/16, K, 16]  output channels tiled by 16, as the QACC kernel reads them
    <base>.ws   f32 [N]   weight scale per output channel
    <base>.b    f32 [N]   bias
    <base>.pre  i32 [chunks, N]  (project only) QACC preload per K chunk that centres the running sum;
                the engine subtracts it again
Activation points: `<point>.max` f32 [1] = calibrated max|x| (int8 scale is max/127; for `.gelu`
points the int8 code q stands for (q + 128) * (max - GELU_MIN) / 255 + GELU_MIN).
Everything else is copied as f32 under its PyTorch name.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quant_eval as qe                                       # noqa: E402
from tiny_voca_src import REPO, WEIGHTS, load_lexicon, phonemize  # noqa: E402
import tvq1                                                   # noqa: E402
from tvq import Golden, weight_int8, conv_as_matrix           # noqa: E402

PROJECT_CHUNKS = 2


def tile16(q):
    """[N, K] int8 -> [Npad/16, K, 16] with zero padding of the output channels."""
    n, k = q.shape
    npad = (n + 15) // 16 * 16
    padded = np.zeros((npad, k), np.int8)
    padded[:n] = q
    return np.ascontiguousarray(padded.reshape(npad // 16, 16, k).transpose(0, 2, 1))


def project_preload(wq, mean_code, chunks):
    """Centre the expected running sum of each K chunk around zero (see the module doc)."""
    n, k = wq.shape
    size = k // chunks
    out = np.zeros((chunks, n), np.int32)
    for c in range(chunks):
        w = wq[:, c * size:(c + 1) * size].astype(np.float64)
        expected = np.concatenate((np.zeros((n, 1)), np.cumsum(w * mean_code[None, c * size:(c + 1) * size], 1)), 1)
        out[c] = np.round(-0.5 * (expected.max(1) + expected.min(1)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(WEIGHTS / "tiny-voca.pt"))
    parser.add_argument("--lexicon", default=str(WEIGHTS / "phonemizer/lexicon.tsv"))
    parser.add_argument("--output", default=str(REPO / "out"))
    parser.add_argument("--test-sentence", type=int, default=0)
    parser.add_argument("--test-voice", type=int, default=1)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    lexicon = load_lexicon(args.lexicon)
    model = qe.load_model(args.checkpoint, check_frontend=False)
    golden = Golden(model)
    calibration = [phonemize(s, lexicon) for s in qe.CALIBRATION]
    with torch.inference_mode():
        golden.calibrate([(p.ids, p.stress) for p in calibration])
        # mean GELU code per input channel of every `project`, for the centring preload
        sums, counts = {}, {}
        for p in calibration:
            for voice in (0, 1):
                golden.synthesize(p.ids, p.stress, voice, dump=True)
                for name, codes in golden.dump.items():
                    if name.endswith(".gelu") and codes.dtype == torch.int8:
                        key = name[:-len(".gelu")] + ".project"
                        sums[key] = sums.get(key, 0) + codes.double().sum(0).numpy()
                        counts[key] = counts.get(key, 0) + codes.shape[0]

    tensors = {}
    state = model.state_dict()
    sd = {k: v.detach().float().numpy() for k, v in state.items()}
    for base in golden.w:
        w = state[base + ".weight"]
        matrix = conv_as_matrix(w) if w.ndim == 3 else w
        q, scale = weight_int8(matrix)
        q = q.numpy().astype(np.int8)
        if q.shape[1] % 32:
            raise ValueError(f"{base}: K={q.shape[1]} is not a multiple of 32")
        tensors[base + ".wq"] = tile16(q)
        tensors[base + ".ws"] = scale.numpy().astype(np.float32)
        tensors[base + ".b"] = sd[base + ".bias"]
        if base.endswith(".project"):
            tensors[base + ".pre"] = project_preload(q, sums[base] / counts[base], PROJECT_CHUNKS)
    for name, value in golden.scales.items():
        tensors[name + ".max"] = np.array([value], np.float32)
    for name, value in sd.items():
        base = name.rsplit(".", 1)[0]
        if base in golden.w and name.endswith((".weight", ".bias")):
            continue
        tensors[name] = value.astype(np.float32)
    size = tvq1.write(out / "tvoca.bin", tensors)
    int8_bytes = sum(v.nbytes for k, v in tensors.items() if k.endswith(".wq"))
    print(f"tvoca.bin: {size / 1e6:.2f} MB ({int8_bytes / 1e6:.2f} MB int8 weights, {len(tensors)} tensors)")

    # ---------------------------------------------------------------- test vector
    p = phonemize(qe.SENTENCES[args.test_sentence], lexicon)
    with torch.inference_mode():
        result = golden.synthesize(p.ids, p.stress, args.test_voice, seed=0, dump=True)
    dump = golden.dump
    test = {"ids": np.array(p.ids, np.int32), "stress": np.array(p.stress, np.int32),
            "voice": np.array([args.test_voice], np.int32), "noise": dump["noise"].numpy(),
            "durations": dump["durations"].numpy().astype(np.int32), "audio": result["audio"].numpy()}
    for name, value in dump.items():
        if name in test or name.endswith(".float") and not name.startswith("encoder.blocks.0"):
            continue
        v = value.detach().numpy()
        test[name] = v.astype(np.int8 if v.dtype == np.int8 else np.int32 if v.dtype.kind in "iu" else np.float32)
    stem = f"test{args.test_sentence}"
    size = tvq1.write(out / f"{stem}.bin", test)
    slim = {k: test[k] for k in ("ids", "stress", "voice", "noise", "durations")}
    tvq1.write(out / f"{stem}_dev.bin", slim)        # what the device needs (flash partition)
    print(f"{stem}.bin: {size / 1e6:.2f} MB, {len(p.ids)} tokens, {int(result['durations'].sum())} frames, "
          f"voice {args.test_voice}: {qe.SENTENCES[args.test_sentence]!r}")


if __name__ == "__main__":
    main()
