---
license: apache-2.0
base_model: n4ze3m/tiny-voca
pipeline_tag: text-to-speech
language:
- en
tags:
- esp32
- esp32-s3
- embedded
- int8
---

# tiny-voca for the ESP32-S3

Prebuilt firmware and model for [tiny-voca-esp32s3](https://github.com/ngxson/tiny-voca-esp32s3):
[tiny-voca](https://huggingface.co/n4ze3m/tiny-voca), a 4.5M-parameter English text-to-speech model,
running entirely on an ESP32-S3 (phonemizer, int8 neural network and speaker).

## Flash

```bash
pip install esptool huggingface_hub
hf download ngxson/tiny-voca-esp32s3 --local-dir out
esptool write-flash 0x0 out/waveshare_s3_audio/firmware.bin 0x210000 out/tvoca.bin 0x910000 out/phonemizer.bin
```

Then open the serial console (115200, or `monitor.sh` from the GitHub repo) and type a sentence.

## Files

| File | Flash offset | What |
|---|---|---|
| `waveshare_s3_audio/firmware.bin` | `0x0` | Bootloader, partition table and app for the Waveshare ESP32-S3-AUDIO-Board |
| `tvoca.bin` | `0x210000` | The model: int8 weights and activations (W8A8), 4.9 MB |
| `phonemizer.bin` | `0x910000` | Data of the dizione phonemizer (CMUdict-based lexicon, unknown-word model) |
| `test0_dev.bin` | `0x810000` | Optional: test vector for the `TEST` console command |

SHA-256: SHA256_TABLE

Built from GitHub commit `COMMIT`.

## License

Apache-2.0. The weights are converted from [tiny-voca](https://huggingface.co/n4ze3m/tiny-voca)
(Apache-2.0); `phonemizer.bin` contains CMUdict and homograph data under the licenses in `licenses/`.
