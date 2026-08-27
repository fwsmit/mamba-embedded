#!/usr/bin/env python3
"""Build a calibration (PTQ) subset of Speech Commands v0.02.

Unlike the augmented training set (in which all non-core words are collapsed
into a single "unknown" class), this uses the *original* dataset so that every
word keeps its own label.  The calibration set contains 20 examples of each of
the 35 words plus 20 silence clips, stored in the same pickle format as the
augmented dataset (SpeechCommandsMFCC).
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from kws_dataset_gen import (
    build_mfcc_transform,
    load_waveform,
    random_noise_clip,
    to_mfcc,
)

N_PER_WORD = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Path to the extracted Speech Commands v0.02 directory")
    parser.add_argument("out", help="Output calib.pkl path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.dataset_root).expanduser().resolve()
    transform = build_mfcc_transform()

    # Every directory is a word (known or auxiliary/silence candidates are
    # background-noise dirs / files, not words).
    words = sorted(d.name for d in root.iterdir()
                   if d.is_dir() and not d.name.startswith("_")
                   and d.name not in ("LICENSE", "README.md"))
    labels = list(words) + ["silence"]
    label2idx = {lbl: i for i, lbl in enumerate(labels)}

    val_set = {l.strip() for l in (root / "validation_list.txt").read_text().splitlines()}
    test_set = {l.strip() for l in (root / "testing_list.txt").read_text().splitlines()}

    X, y = [], []

    for word in tqdm(words, desc="words"):
        word_dir = root / word
        wavs = sorted(f for f in word_dir.glob("*.wav")
                      if f"{word}/{f.name}" not in val_set and f"{word}/{f.name}" not in test_set)
        chosen = random.sample(wavs, N_PER_WORD)
        for f in chosen:
            X.append(to_mfcc(load_waveform(f), transform))
            y.append(label2idx[word])

    # Silence: draw random 1-second windows from the background-noise pool.
    noise_dir = root / "_background_noise_"
    noise_pool = [load_waveform(f) for f in noise_dir.glob("*.wav")]
    if not noise_pool:
        sys.exit("ERROR: no background noise clips found under _background_noise_/")
    idx = label2idx["silence"]
    X_sil, y_sil = zip(*( (to_mfcc(random_noise_clip(noise_pool), transform), idx)
                          for _ in range(N_PER_WORD) ))
    X.extend(list(X_sil))
    y.extend(list(y_sil))

    X = np.stack(X, axis=0).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    perm = np.random.permutation(len(y))

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        import pickle
        pickle.dump(
            {"X": X[perm], "y": y[perm], "label2idx": label2idx,
             "idx2label": {i: l for l, i in label2idx.items()}},
            fh, protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"  ✓ {len(X):>3} samples  →  {out}")
    print(f"    per-word counts: {dict(zip(*np.unique(y, return_counts=True)))}")


if __name__ == "__main__":
    main()