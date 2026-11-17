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

```bash
python esp32/tools/export.py                      # out/tvoca.bin, out/test0.bin, out/test0_dev.bin
make -C esp32/engine/host && esp32/engine/host/tvhost esp32/out/tvoca.bin esp32/out/test0.bin esp32/out/c.wav
esp32/firmware/run.sh --data                      # build, flash app + model + test vector, run the test
python esp32/tools/speak_s3.py --text "Hello there."   # phonemize on the Mac, synthesize on the board
```

ESP-IDF v6.1 (`. ~/work/esp-idf/export.sh`).
