"""Read the firmware's console: print the log, decode TVDUMP base64 blocks into a TVQ1 file.
Hard deadline; stops on TVDUMP_DONE, "tv: finished", a failed test, a panic or a reboot.

    python tools/serial_dump.py /dev/ttyACM0 180 out/dev.bin TEST     # COMMAND is sent once READY
"""
import argparse, base64, signal, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tvq1                                                   # noqa: E402
import tvserial                                               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("seconds", type=float, help="hard deadline")
ap.add_argument("out", help="TVQ1 file for the decoded dumps")
ap.add_argument("command", nargs="?", help="console line to send once the board is READY (e.g. TEST)")
args = ap.parse_args()
send = args.command.encode() + b"\n" if args.command else None


def bail(*_):
    print("\n[serial_dump: hard timeout]")
    sys.exit(2)


signal.signal(signal.SIGALRM, bail)
signal.alarm(max(1, int(args.seconds)))

s = tvserial.open_port(args.port, wait=args.seconds)
if send:
    s.write(b"\n")

dumps, current, chunks, pending = {}, None, [], b""
finished = (b"TVDUMP_DONE", b"tv: finished", b"no test vector", b"synthesize failed")
panic = tuple(m.encode() for m in tvserial.PANIC)
done = panicked = False
while not done:
    try:
        data = s.read(65536)
    except Exception:           # the board went away (reboot): wait for the port to come back
        time.sleep(0.3)
        s = tvserial.open_port(args.port, wait=args.seconds)
        continue
    pending += data
    while b"\n" in pending:
        raw, pending = pending.split(b"\n", 1)
        line = raw.decode(errors="replace").rstrip("\r")
        if send and line.endswith("READY"):
            s.write(send); send = None
            continue
        if line.startswith("TVDUMP_END"):
            name, rows, cols = current or ("?", 0, 0)
            dumps[name] = (base64.b64decode("".join(chunks)), rows, cols)
            print(f"[decoded {name}: {rows}x{cols}]")
            current = None
        elif line.startswith("TVDUMP ") and not line.startswith("TVDUMP_"):
            _, name, rows, cols = line.split()
            current, chunks = (name, int(rows), int(cols)), []
        elif current is not None:
            chunks.append(line)
        else:
            print(line)
        panicked = panicked or any(m in raw for m in panic)
        done = done or panicked or any(m in raw for m in finished)
if panicked:
    time.sleep(1.0)             # the rest of the panic report
    print((pending + s.read(16384)).decode(errors="replace"))

if dumps:
    tvq1.write_raw(args.out, [(name, tvq1.F32, (rows, cols), blob) for name, (blob, rows, cols) in dumps.items()])
    print(f"[wrote {args.out}: {', '.join(dumps)}]")
