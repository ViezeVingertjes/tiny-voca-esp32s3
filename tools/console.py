"""Send lines to the firmware console and print the replies (each line waits for the next READY).
Hard deadline. With --boot, first prints the boot log up to the first READY.

    python tools/console.py --boot "Hello there." "voice F2" "How are you?"
"""
import argparse, signal, sys, time
import serial

ap = argparse.ArgumentParser()
ap.add_argument("lines", nargs="*")
ap.add_argument("--port", default="/dev/cu.usbmodem1201")
ap.add_argument("--boot", action="store_true")
ap.add_argument("--timeout", type=int, default=300)
args = ap.parse_args()
signal.signal(signal.SIGALRM, lambda *_: sys.exit("\n[console: hard timeout]"))
signal.alarm(args.timeout)

s = serial.Serial()
s.port, s.baudrate, s.timeout = args.port, 115200, 0.2
s.dtr = s.rts = False
for _ in range(50):
    try:
        s.open(); break
    except Exception:
        time.sleep(0.2)
buf = b""

def until_ready(echo=True):
    global buf
    while True:
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            line = raw.decode(errors="replace").rstrip("\r")
            if line.startswith("tts> "):
                line = line[5:]
            if line == "READY":
                return
            if echo and not line.startswith(("I (", "W (")):
                print(line, flush=True)
            if "Guru Meditation" in line or "abort()" in line:
                time.sleep(1); print(s.read(8192).decode(errors="replace")); sys.exit(1)
        try:
            buf += s.read(4096)
        except Exception:
            time.sleep(0.3)

if args.boot:
    until_ready()
else:
    s.write(b"\n"); until_ready(echo=False)
for line in args.lines:
    print(f">>> {line}", flush=True)
    t0 = time.time()
    s.write(line.encode() + b"\n")
    until_ready()
    print(f"    ({time.time() - t0:.1f} s)", flush=True)
