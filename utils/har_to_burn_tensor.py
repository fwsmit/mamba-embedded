"""
Transform a HAR UCI dataset sample into a Burn `Tensor<Backend, 3>` compatible format.

Pipeline:
  1. Load the pre-processed HAR feature vectors from the UCI text files.
  2. Apply z-score normalisation using training-set statistics.
  3. Add the batch dimension → shape [1, 561, 1]  (N, D, C).
       C=1 is a dummy channel so the tensor rank matches a 1-D conv / Mamba input.
       Drop the unsqueeze(2) for C if your model expects Tensor<Backend, 2> [N, D].
  4. Export the raw float32 data so Burn can consume it:
       • a .npy file  (load with burn-import / ndarray feature)
       • a .bin file  (load with TensorData::from_bytes)
       • an inline Rust snippet printed to stdout
  5. Save normalisation stats (mean + std) as .npy so the same transform can be
     reproduced at inference time without re-loading the training set.
"""

import numpy as np
import torch


# ── Configuration ─────────────────────────────────────────────────────────────

har_data_dir = "./data/har-uci-dataset/UCI HAR Dataset"
sample_index = 2000
split = "test"  # "train" or "test"


# ── 1. Dataset ────────────────────────────────────────────────────────────────

# Activity names (1-indexed in the label files)
activity_labels: dict[int, str] = {}
with open(f"{har_data_dir}/activity_labels.txt") as fh:
    for line in fh:
        idx, name = line.strip().split()
        activity_labels[int(idx)] = name

X_train = np.loadtxt(f"{har_data_dir}/train/X_train.txt")  # (7352, 561)
y_train = np.loadtxt(f"{har_data_dir}/train/y_train.txt", dtype=int)

print(f"Loaded X_train : {X_train.shape}  {X_train.dtype}")
print(f"Loaded y_train : {y_train.shape}")

# ── 2. Normalisation (z-score, fit on training set only) ─────────────────────

X_mean: np.ndarray = X_train.mean(axis=0)  # (561,)
X_std: np.ndarray = X_train.std(axis=0) + 1e-8  # (561,)  — avoid /0
X_train_norm = (X_train - X_mean) / X_std

# Persist stats so inference code can apply the same transform
np.save("har_norm_mean.npy", X_mean)
np.save("har_norm_std.npy", X_std)
print("Saved normalisation stats → har_norm_mean.npy, har_norm_std.npy")

# Normalise the chosen split with training stats
if split == "train":
    X_split = X_train_norm
    y_split = y_train
else:
    X_test = np.loadtxt(f"{har_data_dir}/test/X_test.txt")  # (2947, 561)
    y_test = np.loadtxt(f"{har_data_dir}/test/y_test.txt", dtype=int)
    X_split = (X_test - X_mean) / X_std
    y_split = y_test
    print(f"Loaded X_test  : {X_test.shape}  {X_test.dtype}")


# ── 3. Pick one sample ────────────────────────────────────────────────────────

sample_features = X_split[sample_index]  # shape: (561,)
label = int(y_split[sample_index])  # 1-indexed (1 … 6)
activity_name = activity_labels[label]

print(f"\nSplit        : {split}")
print(f"Sample index : {sample_index}")
print(f"Label        : {label}  →  {activity_name}")
print(f"Features     : {sample_features.shape}  {sample_features.dtype}")
print(f"min / max    : {sample_features.min():.4f} / {sample_features.max():.4f}")


# ── 4. Convert to PyTorch → Tensor<Backend, 3>  [N, D, C] ────────────────────

# (561,) → (1, 561, 1)  i.e. [N=1, D=561, C=1]
burn_tensor: torch.Tensor = (
    torch.from_numpy(sample_features).float().unsqueeze(0).unsqueeze(2)
)
print(f"\nBurn-ready shape [N,D,C]: {list(burn_tensor.shape)}")


# ── 5a. Export as NumPy (.npy) — usable with burn-import ─────────────────────

# npy_path  = "har_sample.npy"
# np_array  = burn_tensor.numpy()           # shape (1, 561, 1), float32
# np.save(npy_path, np_array)
# print(f"\nSaved .npy → {npy_path}  shape={np_array.shape}  dtype={np_array.dtype}")


# ── 5b. Export as raw little-endian float32 binary ───────────────────────────

# bin_path = "har_sample.bin"
# flat     = burn_tensor.numpy().flatten()
# with open(bin_path, "wb") as fh:
#     fh.write(struct.pack(f"<{len(flat)}f", *flat.tolist()))
# print(f"Saved .bin → {bin_path}  ({len(flat)} float32 values, {len(flat)*4} bytes)")


# ── 5c. Emit a Rust snippet using from_floats (no_std compatible) ─────────────
#
# from_floats() accepts a nested array literal whose nesting depth matches the
# tensor rank.  For Tensor<Backend, 3> with shape [N, D, C] = [1, 561, 1] the
# literal type is [[[f32; 1]; 561]; 1] — three bracket levels.
#
# The 561 feature values are wrapped at COLS_PER_ROW values per source line so
# the generated file stays readable in an editor / diff.

N, D, C = 1, 561, 1
COLS_PER_ROW = 8  # feature values per source line
values_3d = burn_tensor.numpy()  # shape (1, 561, 1)


def fmt(v: float) -> str:
    return f"{v:.6f}_f32"


# Build the feature literal with line-wrapping — iterates over D (561 features),
# each wrapped in its own C=1 inner array: [[val], [val], ...]
chunks: list[str] = []
row_indent = " " * 12
for start in range(0, D, COLS_PER_ROW):
    end = min(start + COLS_PER_ROW, D)
    vals = ", ".join(f"[{fmt(values_3d[0, d, 0])}]" for d in range(start, end))
    chunks.append(f"{row_indent}{vals}")

feature_lit = "[\n" + ",\n".join(chunks) + f"\n{' ' * 8}]"
batch_lit = f"[\n    {feature_lit}\n]"

rust_snippet = f"""// ── Rust / Burn snippet (no_std compatible) ──────────────────────────────────
// Cargo.toml:
//   burn = {{ version = "0.14", default-features = false, features = ["ndarray"] }}
//
// Normalisation applied upstream (z-score, training-set stats):
//   mean saved in har_norm_mean.npy
//   std  saved in har_norm_std.npy
//
// from_floats() takes a nested array literal — no vec!, no alloc needed for
// constructing the literal itself.

use burn::{{
    backend::NdArray,
    tensor::Tensor,
}};

type Backend = NdArray<f32>;

/// HAR feature tensor — shape [N=1, D=561, C=1]
/// Split: {split}, index: {sample_index}, label: {label-1} ({activity_name})
/// Generated by har_to_burn_tensor.py
pub fn input_tensor() -> Tensor<Backend, 3> {{
    Tensor::<Backend, 3>::from_floats(
        {batch_lit},
        &Default::default(),
    )
}}
"""

print(rust_snippet)

rs_path = "src/data/har_tensor.rs"
with open(rs_path, "w") as fh:
    fh.write(rust_snippet.strip())
print(f"Saved Rust snippet → {rs_path}")
