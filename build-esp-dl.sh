#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set +eu
source ~/.espressif/tools/activate_idf_v6.0.1.sh >/dev/null
set -eu

IDF_PY="$IDF_PATH/tools/idf.py"
PYTHON="$IDF_PYTHON_ENV_PATH/bin/python"

cd "$SCRIPT_DIR/esp-dl"

# Capture build+flash output; only print on failure
BUILD_OUT=$("$PYTHON" "$IDF_PY" build 2>&1) || {
  echo "$BUILD_OUT" | awk 'length < 300'
  exit 1
}
"$PYTHON" "$IDF_PY" -p /dev/ttyACM0 flash &>/dev/null

"$PYTHON" - <<'PYEOF'
import serial, sys, time

PORT, BAUD, TIMEOUT_S = "/dev/ttyACM0", 115200, 60
OK_SENTINELS   = ["INFERENCE_OK"]
FAIL_SENTINELS = ["INFERENCE_OOM", "INFERENCE_FAIL"]
CRASH_MARKERS  = ["Guru Meditation Error", "abort() was called", "Backtrace:", "rst:0x"]

exit_code = 3
lines = []

with serial.Serial(PORT, BAUD, timeout=0.1) as ser:
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        raw = ser.readline()
        if not raw: continue
        line = raw.decode("utf-8", errors="replace").rstrip()
        lines.append(line)
        if any(s in line for s in OK_SENTINELS):
            print("\n".join(lines)); exit_code = 0; break
        if any(s in line for s in FAIL_SENTINELS):
            print("\n".join(lines)); exit_code = 1; break
        if any(m in line for m in CRASH_MARKERS):
            for _ in range(20):
                extra = ser.readline()
                if extra: lines.append(extra.decode("utf-8", errors="replace").rstrip())
            print("\n".join(lines)); exit_code = 2; break
    else:
        print(f"TIMEOUT after {TIMEOUT_S}s", file=sys.stderr)

sys.exit(exit_code)
PYEOF
