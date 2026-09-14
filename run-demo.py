#!/usr/bin/env python3
"""Monitor the I2S microphone demo app over serial.

By default this only opens the serial connection and streams the demo's
output; no build or flash happens. With --flash it additionally copies
model.espdl (never the dataset), builds the ESP-DL project with
ACTIVE_APP=demo and flashes /dev/ttyACM0 before monitoring, like run-demo.sh.
It opens a graphical window showing the live MCU information in a dashboard:
the 24-sample waveform stamp, predicted keyword + confidence bar, RMS level
meter, and rolling history plots of prediction, confidence, RMS and inference
latency. Falls back to text-only monitoring when no display is available.
Every newly guessed keyword (different from the previous guess) is published
on a localhost TCP socket (see docs/word-socket.md).

Usage: run-demo.py [--flash] [-q] [--port N] [<path-to-model.espdl>]
  --flash  build and flash before monitoring (requires the model path)
  -q       quiet build/flash output (serial output is always streamed live)
  --port N localhost TCP port for the keyword socket (default 8765)
"""

import os
import re
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

try:
    import serial
except ImportError:
    serial = None

import tkinter as tk

SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_APP = SCRIPT_DIR / "esp-dl/main/apps/demo/app_main.cpp"
MODEL_DEST = SCRIPT_DIR / "esp-dl/main/model/model.espdl"
DATASET_BIN = SCRIPT_DIR / "esp-dl/main/model/dataset.bin"
ESP_DL_DIR = SCRIPT_DIR / "esp-dl"
IDF_ACTIVATE = Path.home() / ".espressif/tools/activate_idf_v6.0.1.sh"
PORT = "/dev/ttyACM0"
BAUD = 115200

FAIL_SENTINELS = ["MIC_TEST_FAIL", "INFERENCE_OOM", "INFERENCE_FAIL"]
CRASH_MARKERS = ["Guru Meditation Error", "abort() was called", "Backtrace:",
                 "rst:0x"]

# Exact printf formats from esp-dl/main/apps/demo/app_main.cpp
RE_BLOCK = re.compile(r"=== KWS block (\d+) \(([\d.]+) s, (\d+) frames\) ===")
RE_STATS = re.compile(
    r"active ([LR]) \| rms\s*(-?inf|[-\d.]+) dBFS \| "
    r"peak\s*(-?inf|[-\d.]+)/\s*(-?inf|[-\d.]+) dBFS \| wave (\S+)")
RE_PRED = re.compile(r"pred (-?\d+)\s+(\S+)\s*\| conf (-?[\d.]+) \| latency ([\d.]+) ms")
RE_LOGITS = re.compile(r"logits((?:\s+[+-]?[\d.]+)+)")

KWS_LABELS = ["down", "go", "left", "no", "off", "on",
              "right", "stop", "up", "yes", "silence", "unknown"]

CLASS_COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
                "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff"]


WORD_SOCKET_PORT = 8765


def usage(stream=sys.stderr):
    print("Usage: run-demo.py [--flash] [-q] [--port N] [<path-to-model.espdl>]",
          file=stream)
    print("  --flash  build and flash before monitoring (requires the model path)",
          file=stream)
    print("  -q       quiet build/flash output (serial output is always streamed live)",
          file=stream)
    print("  --port N localhost TCP port for the keyword socket (default 8765)",
          file=stream)


def parse_args(argv):
    quiet = False
    flash = False
    port = WORD_SOCKET_PORT
    positionals = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-q":
            quiet = True
        elif a == "--flash":
            flash = True
        elif a == "--port":
            if i + 1 >= len(argv):
                print("Error: --port requires a value", file=sys.stderr)
                usage()
                raise SystemExit(1)
            try:
                port = int(argv[i + 1])
            except ValueError:
                port = -1
            if not (0 < port < 65536):
                print(f"Error: invalid port: {argv[i + 1]}", file=sys.stderr)
                raise SystemExit(1)
            i += 1
        elif a in ("-h", "--help"):
            usage(sys.stdout)
            raise SystemExit(0)
        elif a.startswith("-"):
            print(f"Unknown option: {a}", file=sys.stderr)
            usage()
            raise SystemExit(1)
        else:
            positionals.append(a)
        i += 1
    if len(positionals) > 1 or (flash and not positionals):
        usage()
        raise SystemExit(1)
    return quiet, flash, port, positionals[0] if positionals else None


def idf_env():
    if not IDF_ACTIVATE.is_file():
        print(f"Error: IDF activation script not found: {IDF_ACTIVATE}",
              file=sys.stderr)
        raise SystemExit(1)
    out = subprocess.run(["bash", str(IDF_ACTIVATE), "-e"],
                         capture_output=True, text=True, check=True)
    env = dict(os.environ)
    orig_path = env.get("PATH", "")
    for line in out.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    # In -e mode the activation script prints PATH with only the IDF tool
    # directories (the pre-activation system PATH is emitted separately as
    # SYSTEM_PATH). When sourced it prepends those dirs onto the existing
    # PATH instead. Reproduce that here, otherwise system tools (git, sed,
    # basename, ...) are invisible to the build.
    idf_path = env.pop("PATH", "")
    if idf_path and orig_path:
        env["PATH"] = idf_path + ":" + orig_path
    elif idf_path:
        env["PATH"] = idf_path
    for key in ("IDF_PYTHON_ENV_PATH", "IDF_PATH"):
        if key not in env:
            print(f"Error: IDF env variable {key} not set by activation script",
                  file=sys.stderr)
            raise SystemExit(1)
    return env


def run_step(env, args, quiet, cwd):
    proc = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    output = []
    for line in proc.stdout:
        output.append(line)
        if not quiet:
            sys.stdout.write(line)
            sys.stdout.flush()
    proc.wait()
    return proc.returncode, output


def build_and_flash(env, quiet):
    idf = os.path.join(env["IDF_PATH"], "tools/idf.py")
    py = os.path.join(env["IDF_PYTHON_ENV_PATH"], "bin/python")
    common = [py, idf, "-D", "ACTIVE_APP=demo"]
    for step, extra in [("build", ["build"]),
                        ("flash", ["-p", PORT, "flash"])]:
        rc, out = run_step(env, common + extra, quiet, ESP_DL_DIR)
        if rc != 0:
            if quiet:  # mirror the shell's awk 'length < 300' filter
                for line in out:
                    if len(line) < 300:
                        sys.stderr.write(line)
            print(f"demo: {step} failed (exit {rc})", file=sys.stderr)
            raise SystemExit(1)


def prepare_model(model_path):
    model_path = Path(model_path).expanduser()
    if not model_path.is_file():
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        raise SystemExit(1)
    if not DEMO_APP.is_file():
        print("Error: demo app not found (esp-dl/main/apps/demo)", file=sys.stderr)
        raise SystemExit(1)
    shutil.copy(model_path, MODEL_DEST)
    print(f"demo: using model {model_path.name}")
    # A leftover dataset.bin (from a previous run-esp.sh) must not be flashed:
    # the demo never reads the dataset partition. run-esp.sh regenerates it.
    DATASET_BIN.unlink(missing_ok=True)


def wave_values(stamp):
    return [ord(c) - 48 if '0' <= c <= '9'
            else (ord(c) - 48 if ':' <= c <= '@'
                  else (48 - ord(c) if 'A' <= c <= 'Z' else 0))
            for c in stamp]


class WordServer:
    """Line-oriented TCP server on localhost that publishes guessed keywords.

    Each message is the keyword as UTF-8 text followed by a newline
    (see docs/word-socket.md).
    """

    HOST = "127.0.0.1"

    def __init__(self, port):
        self.clients = []
        self.lock = threading.Lock()
        self.last_sent = None
        self.sock = None
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.HOST, port))
            srv.listen(4)
        except OSError as e:
            print(f"demo: keyword socket unavailable on port {port}: {e}",
                  file=sys.stderr)
            return
        self.sock = srv
        threading.Thread(target=self._accept_loop, daemon=True).start()
        print(f"demo: publishing keywords on tcp://{self.HOST}:{port}",
              file=sys.stderr)

    def _accept_loop(self):
        while True:
            try:
                conn, _addr = self.sock.accept()
            except OSError:
                break
            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            with self.lock:
                self.clients.append(conn)

    def send(self, word):
        if self.sock is None:
            return
        with self.lock:
            if word == self.last_sent:
                return
            self.last_sent = word
        data = (word + "\n").encode("utf-8")
        with self.lock:
            alive = []
            for c in self.clients:
                try:
                    c.sendall(data)
                    alive.append(c)
                except OSError:
                    try:
                        c.close()
                    except OSError:
                        pass
            self.clients = alive

    def close(self):
        srv, self.sock = self.sock, None
        with self.lock:
            clients, self.clients = self.clients, []
        for c in clients:
            try:
                c.close()
            except OSError:
                pass
        if srv is not None:
            try:
                srv.close()
            except OSError:
                pass


def meter_color(db):
    if db <= -40:
        return "#34d399"
    if db <= -20:
        return "#fbbf24"
    return "#f87171"


def canvas_px(widget, dim, default):
    try:
        return int(widget[dim].rstrip("c"))
    except (ValueError, tk.TclError):
        return default


class Meter(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, bg="#111418", highlightthickness=0, **kw)
        self.value = float("-inf")
        self.bind("<Configure>", lambda e: self.draw())
        self.draw()

    def set(self, v):
        self.value = v
        self.draw()

    def draw(self):
        self.delete("all")
        w = canvas_px(self, "width", 70)
        h = canvas_px(self, "height", 200)
        top, bottom = 14, h - 4
        v = -80.0 if self.value == float("-inf") else max(-80.0, min(0.0, self.value))
        fy = top + (1 - (v + 80.0) / 80.0) * (bottom - top)
        self.create_rectangle(6, fy, w - 16, bottom, fill=meter_color(v), outline="")
        for db in (0, -20, -40, -60, -80):
            yy = top + (1 - (db + 80.0) / 80.0) * (bottom - top)
            self.create_line(2, yy, w - 12, yy, fill="#2a2f36")
            self.create_text(w - 9, yy, text=f"{db}", anchor="w",
                             fill="#6b7280", font=("Helvetica", 7))


class WaveCanvas(tk.Canvas):
    def __init__(self, master, **kw):
        kw.setdefault("height", 200)
        super().__init__(master, bg="#111418", highlightthickness=0, **kw)
        self.vals = [0.0] * 24
        self.bind("<Configure>", lambda e: self.draw())
        self.draw()

    def set_wave(self, vals):
        self.vals = vals
        self.draw()

    def draw(self):
        self.delete("all")
        w = canvas_px(self, "width", 400)
        h = canvas_px(self, "height", 200)
        base = h * 0.72
        amp = base * 0.8
        pts = []
        steps = 8
        for i in range(len(self.vals) - 1):
            a, b = self.vals[i], self.vals[i + 1]
            for s in range(steps):
                t = s / steps
                v = a + (b - a) * t
                x = (i + t) * w / (len(self.vals) - 1)
                pts.append((x, base - v / 24.0 * amp))
        pts.append((w, h))
        pts.append((0, h))
        if pts:
            self.create_polygon(pts, fill="#0f3d2e", outline="#10b981", width=1.2)
        self.create_line(0, base, w, base, fill="#1f2937")
        self.create_text(w - 4, 10, text="waveform", anchor="e",
                         fill="#9ca3af", font=("Helvetica", 9))


class ConfBar(tk.Canvas):
    def __init__(self, master, width, height):
        super().__init__(master, width=width, height=height, bg="#111418",
                         highlightthickness=0)
        self.frac = 0.0
        self.thr = 0.0
        self.draw()

    def set(self, v):
        self.frac = max(0.0, min(1.0, v))
        self.draw()

    def set_threshold(self, frac):
        self.thr = max(0.0, min(1.0, frac))
        self.draw()

    def draw(self):
        self.delete("all")
        w = canvas_px(self, "width", 260)
        h = canvas_px(self, "height", 14)
        self.create_rectangle(0, 0, w, h, fill="#1a1f26", outline="#2a2f36")
        self.create_rectangle(0, 0, int(w * self.frac), h,
                              fill="#34d399", outline="")
        if 0.0 < self.thr < 1.0:
            tx = int(w * self.thr)
            self.create_line(tx, 1, tx, h - 1, fill="#f59e0b")
        self.create_text(w / 2, h / 2, text=f"{self.frac * 100:.0f}%",
                         fill="#0d1117", font=("Helvetica", 8, "bold"))


class LogitBars(tk.Canvas):
    """Raw (pre-softmax) output scores, one vertical bar per class."""

    def __init__(self, master, labels, **kw):
        kw.setdefault("height", 100)
        super().__init__(master, bg="#111418", highlightthickness=0, **kw)
        self.labels = labels
        self.vals = [0.0] * len(labels)
        self.bind("<Configure>", lambda e: self.draw())
        self.draw()

    def set_logits(self, vals):
        self.vals = list(vals)
        self.draw()

    def draw(self):
        self.delete("all")
        w = canvas_px(self, "width", 560)
        h = canvas_px(self, "height", 100)
        self.create_text(2, 2, text="raw logits", anchor="nw",
                         fill="#9ca3af", font=("Helvetica", 9))
        n = len(self.vals)
        if not n:
            return
        lo, hi = min(self.vals), max(self.vals)
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        pad, gap = 4, 3
        bw = (w - pad * 2 - gap * (n - 1)) / max(n, 1)
        top, bottom = 20, h - 22
        if lo < 0.0 < hi:
            zy = top + (0.0 - lo) / (hi - lo) * (bottom - top)
            self.create_line(0, zy, w, zy, fill="#2a2f36")
        for i, v in enumerate(self.vals):
            x = pad + i * (bw + gap)
            yv = top + (v - lo) / (hi - lo) * (bottom - top)
            color = CLASS_COLORS[i] if v == hi else "#3a4150"
            self.create_rectangle(x, yv, x + bw, bottom, fill=color,
                                  outline="")
            self.create_text(x + bw / 2, h - 8, text=self.labels[i],
                             anchor="n", fill="#9ca3af",
                             font=("Helvetica", 7))


class Plot(tk.Canvas):
    """Small scrolling history plot; y-range may be auto (None)."""

    def __init__(self, master, title, ymin, ymax, yfmt="%.0f", class_rows=False,
                 **kw):
        kw.setdefault("height", 140)
        super().__init__(master, bg="#111418", highlightthickness=1,
                         highlightbackground="#2a2f36", **kw)
        self.title = title
        self.ymin, self.ymax, self.yfmt = ymin, ymax, yfmt
        self.class_rows = class_rows
        self.thr = None
        self.xs, self.ys, self.cs = [], [], []
        self.bind("<Configure>", lambda e: self.redraw())

    def set_threshold(self, thr):
        self.thr = thr
        self.redraw()

    def add(self, x, y, color):
        self.xs.append(x)
        self.ys.append(y)
        self.cs.append(color)
        if len(self.xs) > 500:
            del self.xs[:len(self.xs) - 500]
            del self.ys[:len(self.ys) - 500]
            del self.cs[:len(self.cs) - 500]

    def y_limits(self):
        if self.ymin is not None and self.ymax is not None:
            return self.ymin, self.ymax
        ys = [y for y in self.ys if y == y]
        if not ys:
            return 0.0, 1.0
        lo, hi = min(ys), max(ys)
        if self.ymin is not None:
            return self.ymin, max(hi, self.ymin + 1e-6)
        if self.ymax is not None:
            return min(lo, self.ymax - 1e-6), self.ymax
        pad = (hi - lo) * 0.2 or 1.0
        return lo - pad, hi + pad

    def redraw(self):
        self.delete("all")
        w = canvas_px(self, "width", 300)
        h = canvas_px(self, "height", 140)
        x0, x1, y0, y1 = 44, w - 6, 16, h - 22
        self.create_text((x0 + x1) / 2, 2, text=self.title, fill="#9ca3af",
                         font=("Helvetica", 9))
        ymin, ymax = self.y_limits()
        if ymin == ymax:
            ymax = ymin + 1.0

        def yv(v):
            return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

        for i in range(5):
            v = ymin + (ymax - ymin) * i / 4
            yy = yv(v)
            self.create_line(x0, yy, x1, yy, fill="#1f242b")
            self.create_text(x0 - 4, yy, text=self.yfmt % v, anchor="e",
                             fill="#6b7280", font=("Helvetica", 7))
        if self.thr is not None and self.ymin is not None \
                and self.ymax is not None and self.ymin <= self.thr <= self.ymax:
            self.create_line(x0, yv(self.thr), x1, yv(self.thr),
                             fill="#f59e0b", dash=(4, 3))
        if self.class_rows:
            for c in range(12):
                self.create_line(x0, yv(c), x1, yv(c), fill="#1a1f26")
        if not self.xs:
            return
        n = len(self.xs)
        xspan = max(self.xs) - min(self.xs) or 1

        def xv(x):
            return x0 + (x - min(self.xs)) / xspan * (x1 - x0)

        if self.class_rows:
            pw = max((x1 - x0) / n, 2.0)
            for x, y, c in zip(self.xs, self.ys, self.cs):
                cx = xv(x)
                self.create_line(cx, yv(max(y, 0)), cx, yv(min(y + 1, 12)),
                                 fill=c, width=pw, capstyle="round")
        else:
            for i in range(1, n):
                self.create_line(xv(self.xs[i - 1]), yv(self.ys[i - 1]),
                                 xv(self.xs[i]), yv(self.ys[i]),
                                 fill=self.cs[i], width=1.5)
        self.create_text(x1, y1 + 10, text=f"block {self.xs[-1]:.0f}",
                         anchor="e", fill="#6b7280", font=("Helvetica", 7))


class DemoApp:
    def __init__(self, model_name, word_server=None):
        if serial is None:
            raise RuntimeError("pyserial is required (pip install pyserial)")
        try:
            self.ser = serial.Serial(PORT, BAUD, timeout=0.05)
        except serial.SerialException as e:
            raise RuntimeError(f"cannot open serial port {PORT}: {e}") from e
        self.exit_code = 0
        self.closed = False
        self.last_block = 0
        self.last_pred = None
        self.word_server = word_server if word_server is not None else WordServer(WORD_SOCKET_PORT)
        self.root = tk.Tk()
        self.root.title(f"KWS Demo - {model_name}")
        self.root.configure(bg="#0d1117")
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        self.root.after(50, self.poll_serial)

    def _build_ui(self):
        root = self.root
        header = tk.Frame(root, bg="#0d1117")
        header.pack(fill="x", padx=8, pady=(8, 4))
        left = tk.Frame(header, bg="#0d1117")
        left.pack(side="left")
        self.lbl_kw = tk.Label(left, text="-", font=("Helvetica", 40, "bold"),
                               fg="#e5e7eb", bg="#0d1117")
        self.lbl_kw.pack(anchor="w")
        self.conf_bar = ConfBar(left, 260, 14)
        self.conf_bar.pack(anchor="w", pady=(2, 0))
        right = tk.Frame(header, bg="#0d1117")
        right.pack(side="right")
        self.lbl_active = tk.Label(right, text="active -", font=("Helvetica", 12),
                                   fg="#9ca3af", bg="#0d1117")
        self.lbl_active.pack(anchor="e")
        self.lbl_lat = tk.Label(right, text="- ms", font=("Helvetica", 12),
                                fg="#9ca3af", bg="#0d1117")
        self.lbl_lat.pack(anchor="e")

        self.logit_bars = LogitBars(root, KWS_LABELS)
        self.logit_bars.pack(fill="x", padx=8, pady=2)

        thr_row = tk.Frame(root, bg="#0d1117")
        thr_row.pack(fill="x", padx=8, pady=(0, 2))
        tk.Label(thr_row, text="min conf to show command:",
                 font=("Helvetica", 10), fg="#9ca3af",
                 bg="#0d1117").pack(side="left")
        self.thr_var = tk.DoubleVar(value=20.0)
        self.thr_scale = tk.Scale(thr_row, from_=0, to=100, orient="horizontal",
                                  variable=self.thr_var, resolution=1,
                                  showvalue=True, bg="#111418", fg="#e5e7eb",
                                  troughcolor="#1a1f26", highlightthickness=0,
                                  command=lambda _v: self.on_threshold())
        self.thr_scale.pack(side="left", fill="x", expand=True)

        mid = tk.Frame(root, bg="#0d1117")
        mid.pack(fill="both", expand=True, padx=8)
        self.wave = WaveCanvas(mid)
        self.wave.pack(side="left", fill="both", expand=True)
        self.rms_meter = Meter(mid, width=70, height=200)
        self.rms_meter.pack(side="left", padx=(8, 0))

        plots = tk.Frame(root, bg="#0d1117")
        plots.pack(fill="both", expand=True, padx=8, pady=6)
        self.pred_plot = Plot(plots, "prediction", -0.5, 11.5, yfmt="%d",
                              class_rows=True)
        self.conf_plot = Plot(plots, "confidence", 0, 1.05, yfmt="%.1f")
        self.rms_plot = Plot(plots, "rms (dBFS)", -85, 5, yfmt="%.0f")
        self.lat_plot = Plot(plots, "latency (ms)", 0, None, yfmt="%.0f")
        self.pred_plot.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.conf_plot.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.rms_plot.grid(row=1, column=0, sticky="nsew", padx=(0, 4), pady=(4, 0))
        self.lat_plot.grid(row=1, column=1, sticky="nsew", padx=(4, 0), pady=(4, 0))
        plots.columnconfigure(0, weight=1)
        plots.columnconfigure(1, weight=1)
        plots.rowconfigure(0, weight=1)
        plots.rowconfigure(1, weight=1)
        self.on_threshold()

        status = tk.Frame(root, bg="#0d1117")
        status.pack(fill="x", side="bottom", padx=8, pady=(0, 8))
        self.status_var = tk.StringVar(value=f"monitoring {PORT} - waiting for first block")
        tk.Label(status, textvariable=self.status_var, font=("Helvetica", 10),
                 fg="#9ca3af", bg="#0d1117").pack(side="left")
        tk.Button(status, text="Stop", command=self.stop).pack(side="right")

    def stop(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.ser.close()
        except Exception:
            pass
        try:
            self.root.quit()
        except Exception:
            pass

    def poll_serial(self):
        if self.closed:
            return
        try:
            while True:
                raw = self.ser.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
                self.handle_line(line)
        except serial.SerialException as e:
            self.status_var.set(f"serial error: {e} - retrying")
            self.root.after(1000, self.poll_serial)
            return
        self.root.after(50, self.poll_serial)

    def handle_line(self, line):
        if any(s in line for s in FAIL_SENTINELS):
            print(f"\ndemo: device reported a failure: {line}", file=sys.stderr)
            self.exit_code = 1
            self.root.after(10, self.stop)
            return
        if any(m in line for m in CRASH_MARKERS):
            for _ in range(20):
                try:
                    extra = self.ser.readline()
                except serial.SerialException:
                    break
                if extra:
                    print(extra.decode("utf-8", errors="replace").rstrip())
            print("\ndemo: device crashed", file=sys.stderr)
            self.exit_code = 2
            self.root.after(10, self.stop)
            return
        m = RE_BLOCK.match(line)
        if m:
            self.last_block = int(m.group(1))
            self.status_var.set(
                f"block {m.group(1)} - {m.group(2)} s/frame, {m.group(3)} frames")
            return
        m = RE_STATS.search(line)
        if m:
            def f(s):
                return float(s) if s not in ("-inf", "inf") else float("-inf")
            self.on_stats(m.group(1), f(m.group(2)), f(m.group(3)), f(m.group(4)),
                          wave_values(m.group(5)))
            return
        m = RE_PRED.search(line)
        if m:
            self.on_pred(int(m.group(1)), m.group(2), float(m.group(3)),
                         float(m.group(4)))
            return
        m = RE_LOGITS.search(line)
        if m:
            self.on_logits([float(x) for x in m.group(1).split()])

    def on_stats(self, active, rms, peak_neg, peak_pos, wave):
        self.lbl_active.config(text=f"active {active}")
        self.wave.set_wave(wave)
        self.rms_meter.set(rms)
        self.rms_plot.add(self.last_block, max(rms, -80.0), "#f59e0b")
        self.rms_plot.redraw()

    def on_pred(self, idx, label, conf, latency):
        self.last_pred = (idx, label, conf, latency)
        self.word_server.send(label)
        color = CLASS_COLORS[idx] if 0 <= idx < 12 else "#888888"
        self.conf_bar.set(conf)
        self.lbl_lat.config(text=f"{latency:.1f} ms")
        self.pred_plot.add(self.last_block, idx if idx >= 0 else 0, color)
        self.pred_plot.redraw()
        self.conf_plot.add(self.last_block, max(conf, 0.0), "#34d399")
        self.conf_plot.redraw()
        self.lat_plot.add(self.last_block, latency, "#60a5fa")
        self.lat_plot.redraw()
        self.apply_gate()

    def on_logits(self, vals):
        self.logit_bars.set_logits(vals)

    def on_threshold(self):
        frac = self.thr_var.get() / 100.0
        self.conf_bar.set_threshold(frac)
        self.conf_plot.set_threshold(frac)
        if self.last_pred is not None:
            self.apply_gate()

    def apply_gate(self):
        """Show the command only while confidence is at/above the slider."""
        idx, label, conf, _latency = self.last_pred
        thr = self.thr_var.get() / 100.0
        if conf >= thr:
            self.lbl_kw.config(
                text=label,
                fg=CLASS_COLORS[idx] if 0 <= idx < 12 else "#888888")
        else:
            self.lbl_kw.config(text="…", fg="#4b5563")

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\ndemo: monitoring stopped", file=sys.stderr)
            self.exit_code = 0
        finally:
            if not self.closed:
                self.stop()
            try:
                self.root.destroy()
            except Exception:
                pass
        return self.exit_code


def monitor_text_only(port):
    if serial is None:
        print("Error: pyserial is required (pip install pyserial)",
              file=sys.stderr)
        return 1
    word_server = WordServer(port)
    with serial.Serial(PORT, BAUD, timeout=0.1) as ser:
        try:
            while True:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").rstrip()
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
                m = RE_PRED.search(line)
                if m:
                    word_server.send(m.group(2))
                if any(s in line for s in FAIL_SENTINELS):
                    print("\ndemo: device reported a failure: " + line,
                          file=sys.stderr)
                    return 1
                if any(m in line for m in CRASH_MARKERS):
                    for _ in range(20):
                        extra = ser.readline()
                        if extra:
                            print(extra.decode("utf-8", errors="replace").rstrip())
                    print("\ndemo: device crashed", file=sys.stderr)
                    return 2
        except KeyboardInterrupt:
            print("\ndemo: monitoring stopped", file=sys.stderr)
            return 0


def main(argv):
    quiet, flash, port, model = parse_args(argv)
    if flash:
        prepare_model(model)
        env = idf_env()
        build_and_flash(env, quiet)
        print("demo: flashed, monitoring serial output (Ctrl+C to stop)",
              file=sys.stderr)
    else:
        print("demo: monitoring serial output without flashing "
              "(use --flash to build and flash first; Ctrl+C to stop)",
              file=sys.stderr)
    word_server = WordServer(port)
    try:
        app = DemoApp(Path(model).name if model else "demo",
                      word_server=word_server)
    except tk.TclError as e:
        print(f"demo: no display available ({e}); text-only monitoring",
              file=sys.stderr)
        return monitor_text_only(port)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    try:
        return app.run()
    finally:
        word_server.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))