# Provenance

Copied unchanged from [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp),
`src/community_models/tiny_voca/dizione`, at commit cfc30b04b867c7781ee4881f128902a065a73c8e
(added in PR #1311). It is the C++17 port of [dizione](https://github.com/n4ze3m/dizione),
tiny-voca's phonemizer. License: Apache-2.0 (`LICENSE`, from audio.cpp).

Used by `esp32/tools/dizione_cli.cpp` to phonemize text on the host for the ESP32-S3 engine.

## Local change

`Lexicon` can refer to text it does not own (`Lexicon::external`, `Data::lexicon_view`), so the
ESP32-S3 can read the 3.3 MB `lexicon.tsv` from memory-mapped flash instead of copying it into RAM.
With `lexicon_view` empty the behaviour is the original one. `parse_npz` exposes the g2p.npz reader
for data held in memory. Both are marked "tiny-voca esp32" in the code.
