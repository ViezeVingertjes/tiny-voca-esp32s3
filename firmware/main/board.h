// What the app needs from a board. One implementation per folder in firmware/boards/<name>/,
// selected at build time with -DBOARD=<name>.
#pragma once
#include <stdint.h>

const char *board_name(void);
int board_init(int volume);                 // audio (and display, if any); 0 when audio works
void board_play(int16_t *pcm, int n);       // 24 kHz mono, queued; takes ownership (tv_alloc'd)
int board_set_volume(int volume);           // 0..100; 0 on success
void board_beep(int hz, int ms);
// The text being spoken right now (NULL: nothing). Boards without a screen ignore it.
void board_show_text(const char *text);
