# tiny-voca on the ESP32-S3

A from-scratch inference engine for tiny-voca on the ESP32-S3 (240 MHz dual core, PIE SIMD,
8 MB PSRAM), with int8 weights and activations (W8A8).

| Folder | What |
|---|---|
| `tools/tvq.py` | Golden model: the exact int8 numerics the engine implements (the spec) |
| `tools/quant_eval.py` | Renders and measures quantization variants against fp32 |
| `tools/export.py` | Writes `out/tvoca.bin` (weights in the kernel layout) and test vectors |
| `engine/` | Portable C engine; `engine/host` builds it for the Mac and compares every tensor with the golden model |
| `firmware/` | ESP-IDF firmware: PIE kernels, both cores, model in a flash partition |
| `bench/` | Microbenchmarks of the S3 (SIMD kernels, memory, libm) |
| `third_party/dizione` | C++ port of the dizione phonemizer (from audio.cpp, Apache-2.0) |

## Build and flash (Waveshare ESP32-S3-AUDIO-Board)

```bash
python esp32/tools/export.py                                   # out/tvoca.bin, out/test0*.bin
python esp32/tools/pack_phonemizer.py ../models/tiny-voca/phonemizer esp32/out/phonemizer.bin
cd esp32/firmware && . ~/work/esp-idf/export.sh
idf.py build && idf.py -p /dev/cu.usbmodem1201 flash
esptool --port /dev/cu.usbmodem1201 write-flash \
    0x210000 ../out/tvoca.bin 0x810000 ../out/test0_dev.bin 0x910000 ../out/phonemizer.bin
idf.py -p /dev/cu.usbmodem1201 monitor                         # then type a phrase at "tts>"
```

The data partitions only need writing again when the model or phonemizer data change.

Console commands: any text is spoken; `voice F2` / `voice M2`; `beep`; `prof` (per-op timings);
`TEST` (golden test vector, prints a TVDUMP for `tools/serial_dump.py`); `PHON <text>` (ids only).
On the board everything runs locally: sentence splitting, dizione, synthesis and the speaker.
A sentence that would not fit in PSRAM (about 1,400 frames, 15 s) is split at commas.

## Validation

```bash
make -C esp32/engine/host
esp32/engine/host/tvhost esp32/out/tvoca.bin esp32/out/test0.bin esp32/out/c.wav --dump esp32/out/c.bin
python esp32/tools/serial_dump.py /dev/cu.usbmodem1201 180 esp32/out/dev.bin TEST
python esp32/tools/compare_dump.py esp32/out/c.bin esp32/out/dev.bin       # board vs host
python esp32/tools/compare_dump.py esp32/out/test0.bin esp32/out/dev.bin   # board vs golden model
python esp32/tools/speak_s3.py --text "Hello there."   # Mac phonemizes, board synthesizes, WAV back
```

ESP-IDF v6.1 (`. ~/work/esp-idf/export.sh`).
