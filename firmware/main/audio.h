// Speaker output (Waveshare ESP32-S3-AUDIO-Board: ES8311 + NS4150B amplifier), 24 kHz mono.
#pragma once
#include <stdint.h>
int audio_init(int volume);             // 0 on success; volume 0..100
void audio_play(int16_t *pcm, int n);   // queued; takes ownership of pcm (allocated with tv_alloc)
int audio_pending(void);                // chunks waiting to play
void audio_beep(int hz, int ms);
