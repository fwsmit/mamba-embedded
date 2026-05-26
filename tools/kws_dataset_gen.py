#!/usr/bin/env python3
"""
Offline Augmented Google Speech Commands v2 → MFCC Dataset Generator
======================================================================

Produces three serialised PyTorch-ready splits (train.pkl, val.pkl, test.pkl)
with the following properties:

  • Features : 40 MFCC coefficients extracted with a 40 ms frame / 20 ms hop
                → 49 frames × 40 coeffs = 1 960 features per 1-second clip
  • Train     : every sample duplicated N_AUG (20) times with random
                  – time shift   : uniform ±100 ms
                  – noise mixing : random 1-s clip from _background_noise_
                                   scaled by uniform amplitude ∈ [0, NOISE_MAX_AMP]
  • Val/Test  : no augmentation, MFCC extraction only
  • Silence   : N_SILENCE_TRAIN clips synthesised from _background_noise_ for
                  training; N_SILENCE_EVAL clips each for val and test

Usage
-----
    python kws_dataset_gen.py  /path/to/speech_commands_v0.02  ./kws_data

Loading the result
------------------
    from kws_dataset_gen import SpeechCommandsMFCC
    train_ds = SpeechCommandsMFCC.load("kws_data/train.pkl")
    loader   = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    x, y = next(iter(loader))   # x: (64, 49, 40)  y: (64,)
"""

import argparse
import pickle
import random
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── Audio / MFCC parameters ────────────────────────────────────────────────────
SAMPLE_RATE     = 16_000          # Hz
CLIP_SAMPLES    = 16_000          # 1 second exactly

N_MFCC          = 40
N_MELS          = 40
FRAME_LEN_MS    = 40              # 40 ms  → 640 samples @ 16 kHz
HOP_LEN_MS      = 20              # 20 ms  → 320 samples @ 16 kHz
F_MIN           = 20.0            # Hz
F_MAX           = 4_000.0         # Hz

FRAME_LEN   = int(SAMPLE_RATE * FRAME_LEN_MS / 1000)   # 640
HOP_LEN     = int(SAMPLE_RATE * HOP_LEN_MS   / 1000)   # 320

# With center=False: n_frames = ⌊(16000 − 640) / 320⌋ + 1 = 49
N_FRAMES = (CLIP_SAMPLES - FRAME_LEN) // HOP_LEN + 1   # 49

assert N_FRAMES == 49, f"Unexpected frame count: {N_FRAMES}"
assert N_FRAMES * N_MFCC == 1_960

# ── Augmentation parameters ────────────────────────────────────────────────────
N_AUG           = 20              # augmented copies per training sample
MAX_SHIFT_MS    = 100             # ±100 ms  → ±1600 samples
NOISE_MAX_AMP   = 0.1             # noise amplitude ceiling (uniform [0, MAX])
MAX_SHIFT       = int(SAMPLE_RATE * MAX_SHIFT_MS / 1000)   # 1600

# An amplification by about 20 times for the 10 keywords
N_PER_CLASS = 60500

# ── Label space ────────────────────────────────────────────────────────────────

KNOWN_WORDS: list[str] = sorted(
    ["Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go"]
)
SILENCE_LABEL  = "silence"
UNKNOWN_LABEL  = "unknown"
ALL_LABELS     = KNOWN_WORDS + [SILENCE_LABEL, UNKNOWN_LABEL]
LABEL2IDX      = {lbl: idx for idx, lbl in enumerate(ALL_LABELS)}
IDX2LABEL      = {idx: lbl for lbl, idx in LABEL2IDX.items()}
NUM_CLASSES    = len(ALL_LABELS)   # 37


# ══════════════════════════════════════════════════════════════════════════════
#  Audio helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_mfcc_transform() -> T.MFCC:
    """
    MFCC transform with exact frame geometry:
      win_length = 640  (40 ms)
      hop_length = 320  (20 ms)
      center     = False → n_frames = 49 for a 16 000-sample clip
    """
    return T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_fft": 640,          # 40 ms window
            "win_length": 640,
            "hop_length": 320,     # 20 ms stride
            "n_mels": 40,
            "center": False,
            "power": 2.0,
        },
        log_mels=True,
    )


def load_waveform(path: Path) -> torch.Tensor:
    """Load WAV → mono float32 tensor of exactly 16000 samples."""
    wav, sr = torchaudio.load(str(path))

    # Mix to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    wav = wav.squeeze(0)  # (T,)

    # Pad or trim to exactly 16000 samples
    if wav.shape[0] < SAMPLE_RATE:
        pad_amount = SAMPLE_RATE - wav.shape[0]
        wav = torch.nn.functional.pad(wav, (0, pad_amount))
    else:
        wav = wav[:SAMPLE_RATE]

    return wav


def pad_or_trim(wav: torch.Tensor, length: int = CLIP_SAMPLES) -> torch.Tensor:
    """
    Bring any waveform to exactly `length` samples:
      short clips → zero-pad symmetrically (speech centred)
      long  clips → centre-crop
    """
    n = wav.shape[0]
    if n == length:
        return wav
    if n < length:
        total     = length - n
        pad_left  = total // 2
        pad_right = total - pad_left
        return torch.nn.functional.pad(wav, (pad_left, pad_right))
    start = (n - length) // 2
    return wav[start: start + length]


# ── Augmentation operations ────────────────────────────────────────────────────

def random_time_shift(wav: torch.Tensor) -> torch.Tensor:
    """
    Shift waveform by a random offset in [−MAX_SHIFT, +MAX_SHIFT] samples.
    The vacated region is filled with zeros (no wrap-around).
    """
    shift = random.randint(-MAX_SHIFT, MAX_SHIFT)
    if shift == 0:
        return wav
    out = torch.zeros_like(wav)
    if shift > 0:
        out[shift:] = wav[:-shift]
    else:
        out[:shift] = wav[-shift:]
    return out


def random_noise_clip(noise_pool: list[torch.Tensor]) -> torch.Tensor:
    """Draw a random 1-second window from a randomly chosen noise file."""
    src = random.choice(noise_pool)
    if src.shape[0] > CLIP_SAMPLES:
        start = random.randint(0, src.shape[0] - CLIP_SAMPLES)
        return src[start: start + CLIP_SAMPLES]
    return pad_or_trim(src)


def mix_noise(
    wav: torch.Tensor,
    noise_pool: list[torch.Tensor],
) -> torch.Tensor:
    """
    Add a random noise clip scaled by a uniform amplitude ∈ [0, NOISE_MAX_AMP].
    Peak-normalise the mixture to [−1, 1] to prevent clipping.
    """
    noise  = random_noise_clip(noise_pool)
    amp    = random.uniform(0.0, NOISE_MAX_AMP)
    mixed  = wav + amp * noise
    peak   = mixed.abs().max()
    if peak > 1.0:
        mixed = mixed / peak
    return mixed


def augment(wav: torch.Tensor, noise_pool: list[torch.Tensor]) -> torch.Tensor:
    """Apply time-shift then noise-mixing (order matches training-time pipeline)."""
    wav = random_time_shift(wav)
    wav = mix_noise(wav, noise_pool)
    return wav


# ── Feature extraction ─────────────────────────────────────────────────────────

def to_mfcc(wav: torch.Tensor, transform: T.MFCC) -> np.ndarray:
    """
    Args:
        wav: float32 tensor, shape (CLIP_SAMPLES,)
    Returns:
        float32 ndarray, shape (N_FRAMES, N_MFCC) == (49, 40)
    """
    out = transform(wav.unsqueeze(0))   # (1, N_MFCC, N_FRAMES)
    out = out.squeeze(0).T             # (N_FRAMES, N_MFCC)
    assert out.shape == (N_FRAMES, N_MFCC), \
        f"Unexpected MFCC shape {out.shape}, expected ({N_FRAMES}, {N_MFCC})"
    return out.numpy()


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset file discovery
# ══════════════════════════════════════════════════════════════════════════════

def _read_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def discover_dataset(root: Path) -> tuple[
    list[tuple[Path, str]],   # train
    list[tuple[Path, str]],   # val
    list[tuple[Path, str]],   # test
    list[torch.Tensor],       # noise pool
]:
    """
    Walk the Speech Commands root and partition WAV files according to the
    official validation_list.txt / testing_list.txt manifests.

    Returns lists of (wav_path, label_string) tuples.
    Words absent from KNOWN_WORDS are mapped to UNKNOWN_LABEL.
    """
    val_set  = _read_list(root / "validation_list.txt")
    test_set = _read_list(root / "testing_list.txt")

    train_files: list[tuple[Path, str]] = []
    val_files:   list[tuple[Path, str]] = []
    test_files:  list[tuple[Path, str]] = []
    noise_pool:  list[torch.Tensor]     = []

    for word_dir in sorted(root.iterdir()):
        if not word_dir.is_dir():
            continue
        word = word_dir.name

        # ── Collect background noise ───────────────────────────────────────────
        if word == "_background_noise_":
            for f in sorted(word_dir.glob("*.wav")):
                try:
                    noise_pool.append(load_waveform(f))
                    print(f"  [noise] {f.name:40s}  "
                          f"{noise_pool[-1].shape[0]/SAMPLE_RATE:.1f}s")
                except Exception as exc:
                    print(f"  [WARN] could not load {f}: {exc}", file=sys.stderr)
            continue

        if word.startswith("_"):
            continue   # skip any other meta-directories

        label = word if word in KNOWN_WORDS else UNKNOWN_LABEL

        for wav_path in sorted(word_dir.glob("*.wav")):
            rel   = f"{word}/{wav_path.name}"
            entry = (wav_path, label)
            if rel in test_set:
                test_files.append(entry)
            elif rel in val_set:
                val_files.append(entry)
            else:
                train_files.append(entry)

    return train_files, val_files, test_files, noise_pool


# ══════════════════════════════════════════════════════════════════════════════
#  Split processing
# ══════════════════════════════════════════════════════════════════════════════

def convert_list_to_mfcc(
    files: list[tuple[Path, str]],
    transform: T.MFCC,
    *,
    desc: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X : float32 ndarray, shape (N, 49, 40)
    y : int64   ndarray, shape (N,)
    """
    all_x: list[np.ndarray] = []
    all_y: list[int]        = []
    n_skipped = 0
    all_unknown = defaultdict(list)

    for wav_path, label in tqdm(files, desc=desc, unit="file", leave=False):
        if label in KNOWN_WORDS:
            label_idx = LABEL2IDX[label]
            try:
                wav = load_waveform(wav_path)
            except Exception as exc:
                print(f"\n  [WARN] skipping {wav_path.name}: {exc}", file=sys.stderr)
                n_skipped += 1
                continue

            wav = pad_or_trim(wav)

            all_x.append(to_mfcc(wav, transform))
            all_y.append(label_idx)
        else:
            all_unknown[label].append(wav_path)

    for _ in range(N_PER_CLASS):
        # Make sure every word is added evenly
        label_idx = random.choice(list(all_unknown))
        wav_path = random.choice(all_unknown[label_idx])
        try:
            wav = load_waveform(wav_path)
        except Exception as exc:
            print(f"\n  [WARN] skipping {wav_path.name}: {exc}", file=sys.stderr)
            n_skipped += 1
            continue
        all_x.append(to_mfcc(wav, transform))
        all_y.append(label_idx)

    if n_skipped:
        print(f"  [WARN] skipped {n_skipped} files due to load errors.", file=sys.stderr)

    X = np.stack(all_x, axis=0).astype(np.float32)
    y = np.array(all_y, dtype=np.int64)
    return X, y


def _make_silence_clips(
    noise_pool: list[torch.Tensor],
    transform: T.MFCC,
    n_clips: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesise `n_clips` silence examples by drawing random 1-second windows
    from the noise pool (without speech-signal augmentation).
    """
    idx = LABEL2IDX[SILENCE_LABEL]
    all_x = [to_mfcc(random_noise_clip(noise_pool), transform) for _ in range(n_clips)]
    X = np.stack(all_x, axis=0).astype(np.float32)
    y = np.full(n_clips, idx, dtype=np.int64)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════════

class SpeechCommandsMFCC(Dataset):
    """
    Lightweight wrapper around pre-computed MFCC arrays.

    Each ``__getitem__`` returns ``(mfcc, label)`` where

    * ``mfcc``  – ``FloatTensor`` of shape **(49, 40)**
    * ``label`` – ``LongTensor`` scalar in *[0, NUM_CLASSES)*

    Serialisation
    -------------
    >>> ds = SpeechCommandsMFCC(X, y)
    >>> ds.save("kws_data/train.pkl")
    >>> ds = SpeechCommandsMFCC.load("kws_data/train.pkl")
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        assert X.ndim == 3 and X.shape[1:] == (N_FRAMES, N_MFCC), \
            f"X must be (N, {N_FRAMES}, {N_MFCC}), got {X.shape}"
        self.X = torch.from_numpy(X)   # (N, 49, 40)  float32
        self.y = torch.from_numpy(y)   # (N,)          int64

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]

    # ── I/O ───────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {"X": self.X.numpy(), "y": self.y.numpy(),
                 "label2idx": LABEL2IDX, "idx2label": IDX2LABEL},
                fh, protocol=pickle.HIGHEST_PROTOCOL,
            )
        size_mb = path.stat().st_size / 1_048_576
        print(f"    ✓ {len(self):>7,} samples  →  {path}  ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: str | Path) -> "SpeechCommandsMFCC":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        return cls(data["X"], data["y"])

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def class_counts(self) -> dict[str, int]:
        """Return a {label: count} dictionary."""
        counts: dict[str, int] = {}
        for idx in self.y.tolist():
            lbl = IDX2LABEL[idx]
            counts[lbl] = counts.get(lbl, 0) + 1
        return dict(sorted(counts.items()))

    def __repr__(self) -> str:
        return (
            f"SpeechCommandsMFCC("
            f"n={len(self)}, shape={tuple(self.X.shape[1:])}, "
            f"n_classes={self.y.unique().numel()})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate offline augmented Speech Commands v2 MFCC dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_root",
        help="Path to the extracted Speech Commands v0.02 directory "
             "(must contain testing_list.txt, validation_list.txt, _background_noise_/)",
    )
    parser.add_argument(
        "output_dir",
        help="Directory where train.pkl / val.pkl / test.pkl will be written",
    )
    parser.add_argument(
        "--n-aug", type=int, default=N_AUG,
        help="Number of augmented copies per training sample",
    )
    parser.add_argument(
        "--noise-amp", type=float, default=NOISE_MAX_AMP,
        help="Maximum noise amplitude coefficient (uniform [0, value])",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed for reproducibility",
    )
    args = parser.parse_args()

    # ── Seeding ───────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root   = Path(args.dataset_root).expanduser().resolve()
    outdir = Path(args.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Speech Commands v2  →  Offline MFCC Dataset Generator")
    print("=" * 65)
    print(f"  Dataset root  : {root}")
    print(f"  Output dir    : {outdir}")
    print(f"  MFCC shape    : {N_FRAMES} frames × {N_MFCC} coeffs  "
          f"({FRAME_LEN_MS} ms / {HOP_LEN_MS} ms)")
    print(f"  Augmentations : {args.n_aug}× per train sample")
    print(f"    time shift  : ±{MAX_SHIFT_MS} ms")
    print(f"    noise amp   : [0, {args.noise_amp}]")
    print(f"  Samples per class : {N_PER_CLASS}")
    print(f"  Num classes   : {NUM_CLASSES} ({len(KNOWN_WORDS)} words + silence + unknown)")
    print(f"  Seed          : {args.seed}")
    print()

    if not root.exists():
        sys.exit(f"ERROR: dataset root not found: {root}")

    # ── File discovery ────────────────────────────────────────────────────────
    print("── Scanning dataset ─────────────────────────────────────────")
    train_files, val_files, test_files, noise_pool = discover_dataset(root)

    print(f"\n  Train files : {len(train_files):>7,}")
    print(f"  Val   files : {len(val_files):>7,}")
    print(f"  Test  files : {len(test_files):>7,}")
    print(f"  Noise clips : {len(noise_pool):>7,}")

    if not noise_pool:
        sys.exit("ERROR: no background noise files found. "
                 "Check that _background_noise_/ is present under dataset root.")

    transform = build_mfcc_transform()

    # ── Training split ────────────────────────────────────────────────────────
    # print("[1/3] TRAIN")
    # X_tr, y_tr = _process_split(
    #     train_files, noise_pool, transform,
    #     augment_data=True, n_aug=args.n_aug, desc="  train",
    # )
    # X_sil_tr, y_sil_tr = _make_silence_clips(noise_pool, transform, N_SILENCE_TRAIN)
    # X_tr = np.concatenate([X_tr, X_sil_tr], axis=0)
    # y_tr = np.concatenate([y_tr, y_sil_tr], axis=0)
    # # Shuffle so silence isn't always at the end
    # perm = np.random.permutation(len(y_tr))
    # SpeechCommandsMFCC(X_tr[perm], y_tr[perm]).save(outdir / "train.pkl")

    # ── Validation split ──────────────────────────────────────────────────────
    print(f"\n── [2/3] VAL  (no augmentation) ──────────────────────────────────")
    X_va, y_va = convert_list_to_mfcc(
        val_files, transform,
        desc="  val  "
    )
    N_SILENCE_EVAL = len(X_va) / 11
    X_sil_va, y_sil_va = _make_silence_clips(noise_pool, transform, N_SILENCE_EVAL)
    X_va = np.concatenate([X_va, X_sil_va], axis=0)
    y_va = np.concatenate([y_va, y_sil_va], axis=0)
    SpeechCommandsMFCC(X_va, y_va).save(outdir / "val.pkl")

    # ── Test split ────────────────────────────────────────────────────────────
    print(f"\n── [3/3] TEST  (no augmentation) ─────────────────────────────────")
    X_te, y_te = convert_list_to_mfcc(
        test_files, transform,
        desc="  test ",
    )
    N_SILENCE_TEST = len(X_te) / 11
    X_sil_te, y_sil_te = _make_silence_clips(noise_pool, transform, N_SILENCE_TEST)
    X_te = np.concatenate([X_te, X_sil_te], axis=0)
    y_te = np.concatenate([y_te, y_sil_te], axis=0)
    SpeechCommandsMFCC(X_te, y_te).save(outdir / "test.pkl")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary ───────────────────────────────────────────────────────")
    for tag, X, y in [("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)]:
        uniq, cnts = np.unique(y, return_counts=True)
        print(f"  {tag:<5s}  samples={len(y):>8,}  "
              f"X shape={X.shape}  "
              f"classes={len(uniq)}")
    print()
    print("Done. Load with:")
    print("    from kws_dataset_gen import SpeechCommandsMFCC")
    print("    train_ds = SpeechCommandsMFCC.load('kws_data/train.pkl')")


if __name__ == "__main__":
    main()
