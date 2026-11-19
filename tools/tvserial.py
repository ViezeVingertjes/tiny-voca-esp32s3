"""The board's USB serial console, shared by console.py, serial_dump.py and speak_s3.py.
Standard library and pyserial only (serial_dump.py runs under ESP-IDF's Python)."""
import glob, sys, time
import serial

# Lines after which the board is gone (it prints the rest of the panic, then reboots).
PANIC = ("Guru Meditation", "abort() was called", "Rebooting...")
ESPRESSIF_VID = 0x303A          # the S3's built-in USB Serial/JTAG


def default_port():
    """The first Espressif USB device, else the first /dev/cu.usbmodem* (macOS) or /dev/ttyACM* (Linux)."""
    try:
        from serial.tools import list_ports
        for p in sorted(list_ports.comports(), key=lambda p: p.device):
            if p.vid == ESPRESSIF_VID:
                return p.device
    except Exception:
        pass
    for pattern in ("/dev/cu.usbmodem*", "/dev/ttyACM*"):
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]
    return "/dev/ttyACM0"


def open_port(port, wait=10.0, timeout=0.2):
    """Opens the console without resetting the board, retrying for `wait` seconds (the port comes
    and goes while the board reboots). Only RTS is lowered: DTR low with RTS high resets the S3."""
    deadline = time.time() + wait
    while True:
        s = serial.Serial()
        s.port, s.baudrate, s.timeout = port, 115200, timeout
        s.rts = False
        try:
            s.open()
            return s
        except serial.SerialException as e:
            if time.time() > deadline:
                sys.exit(f"cannot open {port}: {e}")
            time.sleep(0.2)


def reset(s):
    """Reboots the board (the USB Serial/JTAG reset sequence); the port must be opened again after."""
    s.dtr = False
    s.rts = True
    time.sleep(0.1)
    s.rts = False
    s.close()
    time.sleep(0.5)
