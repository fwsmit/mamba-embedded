#!/bin/bash
set -euo pipefail

VERBOSE=false

while getopts ":v" opt; do
  case $opt in
  v) VERBOSE=true ;;
  \?)
    echo "Usage: $0 [-v] <path-to-model.espdl>" >&2
    exit 1
    ;;
  esac
done
shift $((OPTIND - 1))

if [ $# -ne 1 ]; then
  echo "Usage: $0 [-v] <path-to-model.espdl>" >&2
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

# Copy the matching dataset-trial-<N>.bin as dataset.bin.
# dataset.bin will be auto-flashed by idf.py flash via the CMake build system.
MODEL_BASENAME=$(basename "$MODEL_PATH")
if [[ "$MODEL_BASENAME" =~ trial-([0-9]+)\.espdl$ ]]; then
  TRIAL_NUM="${BASH_REMATCH[1]}"
  DATASET_SRC="$(dirname "$MODEL_PATH")/dataset-trial-${TRIAL_NUM}.bin"
  if [ ! -f "$DATASET_SRC" ]; then
    echo "Error: dataset file not found: $DATASET_SRC" >&2
    exit 1
  fi
  cp "$DATASET_SRC" "$SCRIPT_DIR/esp-dl/main/model/dataset.bin"
else
  echo "Error: model filename does not contain a trial number (expected *-trial-N.espdl): $MODEL_BASENAME" >&2
  exit 1
fi

cd "$SCRIPT_DIR/esp-dl"

if $VERBOSE; then
  "$PYTHON" "$IDF_PY" build 2>&1 || exit 1
  "$PYTHON" "$IDF_PY" -p /dev/ttyACM0 flash 2>&1 || exit 1
else
  # Capture build output; only print on failure
  BUILD_OUT=$("$PYTHON" "$IDF_PY" build 2>&1) || {
    echo "$BUILD_OUT" | awk 'length < 300'
    exit 1
  }

  # Flash — surfaces errors instead of swallowing them.
  # dataset.bin (if present) is now auto-flashed via the CMake build system.
  FLASH_OUT=$("$PYTHON" "$IDF_PY" -p /dev/ttyACM0 flash 2>&1) || {
    echo "$FLASH_OUT" | awk 'length < 300'
    exit 1
  }
fi

# Monitor serial output — run as a subprocess so stdout stays on bash's pipe
"$PYTHON" - /dev/ttyACM0 "$VERBOSE" <<'PYEOF'
import serial, sys, time

PORT     = sys.argv[1]
VERBOSE  = len(sys.argv) > 2 and sys.argv[2] == "true"
BAUD     = 115200
TIMEOUT_S = 600
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
        if VERBOSE:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        lines.append(line)
        if any(s in line for s in OK_SENTINELS):
            if not VERBOSE:
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()
            exit_code = 0
            break
        if any(s in line for s in FAIL_SENTINELS):
            if not VERBOSE:
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()
            exit_code = 1
            break
        if any(m in line for m in CRASH_MARKERS):
            for _ in range(20):
                extra = ser.readline()
                if extra:
                    extra_line = extra.decode("utf-8", errors="replace").rstrip()
                    if VERBOSE:
                        sys.stdout.write(extra_line + "\n")
                        sys.stdout.flush()
                    lines.append(extra_line)
            if not VERBOSE:
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()
            exit_code = 2
            break
    else:
        sys.stderr.write(f"TIMEOUT after {TIMEOUT_S}s\n")
        sys.stderr.flush()

sys.exit(exit_code)
PYEOF
