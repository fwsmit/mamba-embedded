#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

# ── Audio / MFCC parameters ────────────────────────────────────────────────────
SAMPLE_RATE = 16_000  # Hz
CLIP_SAMPLES = 16_000  # 1 second exactly

N_MFCC = 40
N_MELS = 40
FRAME_LEN_MS = 40  # 40 ms  → 640 samples @ 16 kHz
HOP_LEN_MS = 20  # 20 ms  → 320 samples @ 16 kHz
F_MIN = 20.0  # Hz
F_MAX = 4_000.0  # Hz

FRAME_LEN = int(SAMPLE_RATE * FRAME_LEN_MS / 1000)  # 640
HOP_LEN = int(SAMPLE_RATE * HOP_LEN_MS / 1000)  # 320

# With center=False: n_frames = ⌊(16000 − 640) / 320⌋ + 1 = 49
N_FRAMES = (CLIP_SAMPLES - FRAME_LEN) // HOP_LEN + 1   # 49

assert N_FRAMES == 49, f"Unexpected frame count: {N_FRAMES}"
assert N_FRAMES * N_MFCC == 1_960

# ── Augmentation parameters ────────────────────────────────────────────────────
N_PER_CLASS = 60500
MAX_SHIFT_MS = 100  # ±100 ms  → ±1600 samples
NOISE_MAX_AMP = 0.1  # noise amplitude ceiling (uniform [0, MAX])
MAX_SHIFT = int(SAMPLE_RATE * MAX_SHIFT_MS / 1000)  # 1600


# ── Label space ────────────────────────────────────────────────────────────────

KNOWN_WORDS: list[str] = sorted(
    ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
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


def augment_wav(wav: torch.Tensor, noise_pool) -> torch.Tensor:
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
    all_unknown = defaultdict(list)


    for wav_path, label in tqdm(files, desc=desc, unit="file", leave=False):
        if label in KNOWN_WORDS:
            label_idx = LABEL2IDX[label]
            wav = load_waveform(wav_path)

            all_x.append(to_mfcc(wav, transform))
            all_y.append(label_idx)
        else:
            all_unknown[label].append(wav_path)

    for _ in range(len(all_x)//10):
        # Make sure every word is added evenly
        label = random.choice(list(all_unknown))
        label_idx = LABEL2IDX[UNKNOWN_LABEL]
        wav_path = random.choice(all_unknown[label])
        wav = load_waveform(wav_path)
        all_x.append(to_mfcc(wav, transform))
        all_y.append(label_idx)


    X = np.stack(all_x, axis=0).astype(np.float32)
    y = np.array(all_y, dtype=np.int64)
    return X, y


def convert_list_to_mfcc_augmented(
    files: list[tuple[Path, str]],
    transform: T.MFCC,
    n_per_class,
    *,
    desc: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X : float32 ndarray, shape (N, 49, 40)
    y : int64   ndarray, shape (N,)
    """
    # all_x: list[np.ndarray] = []
    # all_y: list[int] = []
    all_unknown = defaultdict(list)
    all_known = defaultdict(list)

    for wav_path, label in files:
        if label in KNOWN_WORDS:
            all_known[label].append(wav_path)
        else:
            all_unknown[label].append(wav_path)

    n_known = n_per_class * len(KNOWN_WORDS)
    n_unknown = n_known // 10
    N = n_known + n_unknown

    X = np.empty((N, 49, 40), dtype=np.float32)
    y = np.empty((N,), dtype=np.int64)
    idx = 0

    for _ in tqdm(range(n_known), desc=desc, unit="file", leave=False):
        label = random.choice(list(all_known))
        wav = load_waveform(random.choice(all_known[label]))
        X[idx] = to_mfcc(wav, transform)
        y[idx] = LABEL2IDX[label]
        idx += 1

    for _ in range(n_unknown):
        label = random.choice(list(all_unknown))
        wav = load_waveform(random.choice(all_unknown[label]))
        X[idx] = to_mfcc(wav, transform)
        y[idx] = LABEL2IDX[UNKNOWN_LABEL]
        idx += 1
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
    print("[1/3] TRAIN")
    X_tr, y_tr = convert_list_to_mfcc_augmented(
        train_files, transform, n_per_class=N_PER_CLASS,
        desc="  train"
    )
    N_SILENCE_TRAIN = len(X_tr) // 11
    X_sil_tr, y_sil_tr = _make_silence_clips(noise_pool, transform, N_SILENCE_TRAIN)
    X_tr = np.concatenate([X_tr, X_sil_tr], axis=0)
    y_tr = np.concatenate([y_tr, y_sil_tr], axis=0)
    # Shuffle so silence isn't always at the end
    perm = np.random.permutation(len(y_tr))
    print(f"Generated training dataset with {len(X_tr)} samples")
    SpeechCommandsMFCC(X_tr[perm], y_tr[perm]).save(outdir / "train.pkl")

    # ── Validation split ──────────────────────────────────────────────────────
    print(f"\n── [2/3] VAL  (no augmentation) ──────────────────────────────────")
    X_va, y_va = convert_list_to_mfcc(
        val_files, transform,
        desc="  val  "
    )
    N_SILENCE_EVAL = len(X_va) // 11
    X_sil_va, y_sil_va = _make_silence_clips(noise_pool, transform, N_SILENCE_EVAL)
    X_va = np.concatenate([X_va, X_sil_va], axis=0)
    y_va = np.concatenate([y_va, y_sil_va], axis=0)
    print(f"Generated validation dataset with {len(X_va)} samples")
    SpeechCommandsMFCC(X_va, y_va).save(outdir / "val.pkl")

    # ── Test split ────────────────────────────────────────────────────────────
    print(f"\n── [3/3] TEST  (no augmentation) ─────────────────────────────────")
    X_te, y_te = convert_list_to_mfcc(
        test_files, transform,
        desc="  test ",
    )
    N_SILENCE_TEST = len(X_te) // 11
    X_sil_te, y_sil_te = _make_silence_clips(noise_pool, transform, N_SILENCE_TEST)
    X_te = np.concatenate([X_te, X_sil_te], axis=0)
    y_te = np.concatenate([y_te, y_sil_te], axis=0)
    print(f"Generated test dataset with {len(X_te)} samples")
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
