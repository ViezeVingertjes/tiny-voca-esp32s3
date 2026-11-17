#!/bin/bash
# Build, flash (app; model + test vector only with --data), run and capture. Hard timeouts everywhere.
#   ./run.sh [--data] [capture seconds]
set -o pipefail
to() { local s=$1; shift; perl -e 'alarm shift; exec @ARGV' "$s" "$@"; }
PORT=/dev/cu.usbmodem1201
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/../out
DATA=0; [ "$1" = "--data" ] && { DATA=1; shift; }
CAPTURE=${1:-180}
cd "$HERE" || exit 1
. ~/work/esp-idf/export.sh >/dev/null 2>&1
to 400 idf.py build 2>&1 | grep -E "error:|Project build complete" || exit 1
to 180 idf.py -p $PORT flash 2>&1 | grep -E "error|Hash of data verified|Done" | tail -2
if [ $DATA = 1 ]; then
  to 300 esptool --port $PORT write-flash 0x210000 "$OUT/tvoca.bin" 0x810000 "$OUT/test0_dev.bin" 2>&1 | grep -E "Wrote|error|Hash" | tail -3
fi
to $((CAPTURE + 10)) python -I "$HERE/../tools/serial_dump.py" $PORT "$CAPTURE" "$OUT/dev_test0.bin" | grep -av "^I (\|^W (" | sed -n '/####/,$p'
