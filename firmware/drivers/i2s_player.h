// I2S output (standard mode, 16-bit mono, MCLK = 256 x fs) fed by a queue and a player task.
#pragma once
#include <stdint.h>
typedef struct {
    int mclk, bclk, ws, dout;   // GPIO numbers
    int sample_rate;
} i2s_player_pins;

int i2s_player_start(const i2s_player_pins *pins);     // 0 on success; MCLK runs afterwards
void i2s_player_play(int16_t *pcm, int n);             // queued; takes ownership (tv_alloc'd)
int i2s_player_pending(void);                          // chunks waiting
void i2s_player_beep(int hz, int ms);
