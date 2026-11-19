"""Distances between two renders of the same durations: log-mel L1, F0 error, voicing flips.
Needs torch only (no tiny-voca checkout), so compare_dump.py runs anywhere."""
import math
import torch


def log_mel(audio, sample_rate=24000):
    spec = torch.stft(audio, 1024, 256, window=torch.hann_window(1024), return_complex=True).abs() ** 2
    freqs = torch.linspace(0, sample_rate / 2, spec.shape[0])
    mel = 2595 * torch.log10(1 + freqs / 700)
    edges = torch.linspace(0, float(mel[-1]), 82)
    bank = torch.clamp(torch.minimum((mel[None] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None]),
                                     (edges[2:, None] - mel[None]) / (edges[2:, None] - edges[1:-1, None])), min=0)
    return 10 * torch.log10(bank @ spec + 1e-8)


def compare(ref, test):
    """ref/test: dicts with audio, prosody (frames x 3) from the same durations."""
    a, b = ref["audio"], test["audio"]
    ma, mb = log_mel(a), log_mel(b)
    loud = ma.max(0).values > ma.max() - 60                 # skip silent frames
    mel_db = float((ma - mb).abs()[:, loud].mean())
    pa, pb = ref["prosody"], test["prosody"]
    voiced = (pa[:, 1] > 0) & (pb[:, 1] > 0)
    cents = float(((pa[voiced, 0] - pb[voiced, 0]) * 1200 / math.log(2)).pow(2).mean().sqrt())
    flips = float(((pa[:, 1] > 0) != (pb[:, 1] > 0)).float().mean())
    return mel_db, cents, flips
