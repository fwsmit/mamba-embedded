#!/bin/bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <path-to-model.espdl>" >&2
  exit 1
fi

MODEL_PATH="$1"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: model file not found: $MODEL_PATH" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set +eu
source ~/.espressif/tools/activate_idf_v6.0.1.sh >/dev/null
set -eu

IDF_PY="$IDF_PATH/tools/idf.py"
PYTHON="$IDF_PYTHON_ENV_PATH/bin/python"

cp "$MODEL_PATH" "$SCRIPT_DIR/esp-dl/main/model/model.espdl"

# Copy dataset.bin from the same directory, if present
DATASET_SRC="$(dirname "$MODEL_PATH")/dataset.bin"
if [ -f "$DATASET_SRC" ]; then
    cp "$DATASET_SRC" "$SCRIPT_DIR/esp-dl/main/model/dataset.bin"
fi

cd "$SCRIPT_DIR/esp-dl"

# Capture build output; only print on failure
BUILD_OUT=$("$PYTHON" "$IDF_PY" build 2>&1) || {
  echo "$BUILD_OUT" | awk 'length < 300'
  exit 1
}

# Flash — surface errors instead of swallowing them
FLASH_OUT=$("$PYTHON" "$IDF_PY" -p /dev/ttyACM0 flash 2>&1) || {
  echo "$FLASH_OUT" | awk 'length < 300'
  exit 1
}

# Flash dataset to its partition, if present
DATASET_BIN="$SCRIPT_DIR/esp-dl/main/model/dataset.bin"
if [ -f "$DATASET_BIN" ]; then
    DATASET_OFFSET=0x7e0000
    "$PYTHON" -m esptool --chip esp32s3 -p /dev/ttyACM0 -b 460800 \
        --before default_reset --after no_reset \
        write_flash "$DATASET_OFFSET" "$DATASET_BIN" 2>&1
fi

# Monitor serial output — run as a subprocess so stdout stays on bash's pipe
"$PYTHON" - /dev/ttyACM0 <<'PYEOF'
import serial, sys, time

PORT     = sys.argv[1]
BAUD     = 115200
TIMEOUT_S = 60
OK_SENTINELS   = ["INFERENCE_OK"]
FAIL_SENTINELS = ["INFERENCE_OOM", "INFERENCE_FAIL"]
CRASH_MARKERS  = ["Guru Meditation Error", "abort() was called", "Backtrace:", "rst:0x"]

exit_code = 3
lines = []

with serial.Serial(PORT, BAUD, timeout=0.1) as ser:
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").rstrip()
        lines.append(line)
        if any(s in line for s in OK_SENTINELS):
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()
            exit_code = 0
            break
        if any(s in line for s in FAIL_SENTINELS):
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()
            exit_code = 1
            break
        if any(m in line for m in CRASH_MARKERS):
            for _ in range(20):
                extra = ser.readline()
                if extra:
                    lines.append(extra.decode("utf-8", errors="replace").rstrip())
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()
            exit_code = 2
            break
    else:
        sys.stderr.write(f"TIMEOUT after {TIMEOUT_S}s\n")
        sys.stderr.flush()

sys.exit(exit_code)
PYEOF

