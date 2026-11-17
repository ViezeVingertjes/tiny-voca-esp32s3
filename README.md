# tiny-voca on the ESP32-S3

Type a sentence, hear it. [tiny-voca](https://github.com/n4ze3m/tiny-voca), a 4.5M-parameter English
text-to-speech model, running entirely on an ESP32-S3: phonemizer, neural network and speaker.
No cloud, no computer once it is flashed.

## What you need

- A [Waveshare ESP32-S3-AUDIO-Board](https://www.waveshare.com/wiki/ESP32-S3-AUDIO-Board) (16 MB flash, 8 MB PSRAM, speaker)
- [ESP-IDF](https://docs.espressif.com/projects/esp-idf/) v6.1
- To export the model once: a [tiny-voca](https://github.com/n4ze3m/tiny-voca) checkout with this repo
  at `esp32/`, Python with PyTorch, and the weights from [Hugging Face](https://huggingface.co/n4ze3m/tiny-voca)

## Setup

**Step 1: Export the model.** From the tiny-voca checkout:

```bash
git submodule add git@github.com:ngxson/tiny-voca-esp32s3.git esp32
hf download n4ze3m/tiny-voca --local-dir ../models/tiny-voca
python esp32/tools/export.py
python esp32/tools/pack_phonemizer.py ../models/tiny-voca/phonemizer esp32/out/phonemizer.bin
```

**Step 2: Flash.**

```bash
cd esp32/firmware
. $IDF_PATH/export.sh
idf.py build flash
esptool write-flash 0x210000 ../out/tvoca.bin 0x810000 ../out/test0_dev.bin 0x910000 ../out/phonemizer.bin
```

The second line (model and phonemizer data) is only needed again when those files change.

**Step 3: Talk.**

```bash
../monitor.sh
```

```
tts> Hello! This is tiny voca, speaking from the little board on your desk.
```

## Commands

| Command | What it does |
|---|---|
| any text | Speak it |
| `voice F2` / `voice M2` | Female / male voice |
| `vol 50` | Speaker volume, 0 to 100 |
| `beep` | Test the speaker |
| `prof` | Print where the time goes after each sentence |

A short sentence starts playing after about a second, a long one after a few seconds. One sentence
can be up to about 15 seconds long; longer ones are split at commas.

Technical details are in [AGENTS.md](AGENTS.md).

## License

Apache-2.0, see [LICENSE](LICENSE). `third_party/dizione` is from
[audio.cpp](https://github.com/0xShug0/audio.cpp) (Apache-2.0). The model weights come with tiny-voca's license.
