#!/bin/bash
#
# Run every quantized Mamba-Lite model (.espdl) in experiments/mambalite-micro/
# on the ESP32-S3 and record ONLY the MCU serial output (never build/flash
# compiler output) into a per-model <model>.output file next to the model.
#
# Resumable: models that already have an <model>.output file are skipped, so if
# the script is interrupted (Ctrl-C) it continues where it left off on the next
# run. Failed models are kept going over; pass --force to re-run everything.
# New .espdl models added to the directory are picked up automatically.
#
# Usage:
#   ./run-mambalite-mcu.sh [--force] [--verbose]
#
# Environment:
#   PORT   serial device (default: /dev/ttyACM0)
#

set -euo pipefail

VERBOSE=false
FORCE=false
while getopts ":vf" opt; do
  case $opt in
  v) VERBOSE=true ;;
  f) FORCE=true ;;
  \?)
    echo "Usage: $0 [-v] [-f]" >&2
    exit 1
    ;;
  esac
done
shift $((OPTIND - 1))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/experiments/mambalite-micro"
PORT="${PORT:-/dev/ttyACM0}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "Error: model directory not found: $MODEL_DIR" >&2
  exit 1
fi

set +eu
source ~/.espressif/tools/activate_idf_v6.0.1.sh >/dev/null
set -eu

IDF_PY="$IDF_PATH/tools/idf.py"
PYTHON="$IDF_PYTHON_ENV_PATH/bin/python"

# Collect .espdl models, keeping only those with a matching
# dataset-<stem>.bin next to them (same convention as run-esp.sh).
MODELS=()
for m in "$MODEL_DIR"/*.espdl; do
  [ -e "$m" ] || continue
  stem="$(basename "${m%.espdl}")"
  if [ -f "$MODEL_DIR/dataset-$stem.bin" ]; then
    MODELS+=("$m")
  fi
done

if [ ${#MODELS[@]} -eq 0 ]; then
  echo "No .espdl models with a matching dataset-<stem>.bin in $MODEL_DIR" >&2
  exit 1
fi

SKIPPED=()
RUN=()
FAILED=()

for espdl in "${MODELS[@]}"; do
  stem="$(basename "${espdl%.espdl}")"
  out_path="$MODEL_DIR/$stem.output"

  if [ -f "$out_path" ] && ! $FORCE; then
    echo "[skip  ] $stem (output exists: $(basename "$out_path"))"
    SKIPPED+=("$stem")
    continue
  fi

  # Copy model + dataset (firmware embeds model.espdl; CMake auto-flashes
  # dataset.bin into the dataset partition).
  cp "$espdl" "$SCRIPT_DIR/esp-dl/main/model/model.espdl"
  cp "$MODEL_DIR/dataset-$stem.bin" "$SCRIPT_DIR/esp-dl/main/model/dataset.bin"

  # Build & flash (run inside the esp-dl project dir). Compiler output is
  # captured and only printed on failure; it never goes into the .output file.
  echo "[build ] $stem"
  BUILD_OUT=$("$PYTHON" "$IDF_PY" -C "$SCRIPT_DIR/esp-dl" build 2>&1) || {
    echo "  BUILD FAILED:" >&2
    echo "$BUILD_OUT" | tail -n 40 >&2
    FAILED+=("$stem")
    continue
  }

  echo "[flash ] $stem"
  FLASH_OUT=$("$PYTHON" "$IDF_PY" -C "$SCRIPT_DIR/esp-dl" -p "$PORT" flash 2>&1) || {
    echo "  FLASH FAILED:" >&2
    echo "$FLASH_OUT" | tail -n 40 >&2
    FAILED+=("$stem")
    continue
  }

  # Monitor serial output. Writes only MCU output to the .output file;
  # reads the boot log from right after the flash reset.
  echo "[monitor] $stem -> $(basename "$out_path")"
  if ! "$PYTHON" - "$PORT" "$out_path" "$VERBOSE" <<'PYEOF'; then
import serial, sys, time

PORT     = sys.argv[1]
OUT_PATH = sys.argv[2]
VERBOSE  = len(sys.argv) > 3 and sys.argv[3] == "true"
BAUD     = 115200
TIMEOUT_S = 3000
OK_SENTINELS   = ["INFERENCE_OK"]
FAIL_SENTINELS = ["INFERENCE_OOM", "INFERENCE_FAIL"]
CRASH_MARKERS  = ["Guru Meditation Error", "abort() was called", "Backtrace:", "rst:0x"]

exit_code = 3

with open(OUT_PATH, "w") as fh:
    with serial.Serial(PORT, BAUD, timeout=0.1) as ser:
        deadline = time.monotonic() + TIMEOUT_S
        while time.monotonic() < deadline:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").rstrip()
            fh.write(line + "\n")
            fh.flush()
            if VERBOSE:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            if any(s in line for s in OK_SENTINELS):
                exit_code = 0
                break
            if any(s in line for s in FAIL_SENTINELS):
                exit_code = 1
                break
            if any(m in line for m in CRASH_MARKERS):
                for _ in range(20):
                    extra = ser.readline()
                    if extra:
                        extra_line = extra.decode("utf-8", errors="replace").rstrip()
                        fh.write(extra_line + "\n")
                        fh.flush()
                        if VERBOSE:
                            sys.stdout.write(extra_line + "\n")
                            sys.stdout.flush()
                exit_code = 2
                break
        else:
            sys.stderr.write(f"TIMEOUT after {TIMEOUT_S}s\n")
            sys.stderr.flush()

sys.exit(exit_code)
PYEOF
    echo "  MONITOR FAILED for $stem (exit $?)" >&2
    FAILED+=("$stem")
    continue
  fi

  RUN+=("$stem")
done

echo
echo "=== SUMMARY ==="
echo "  ran:     ${#RUN[@]}     ${RUN[*]:-}"
echo "  skipped: ${#SKIPPED[@]} ${SKIPPED[*]:-}"
echo "  failed:  ${#FAILED[@]}  ${FAILED[*]:-}"

if [ ${#FAILED[@]} -gt 0 ]; then
  exit 1
fi