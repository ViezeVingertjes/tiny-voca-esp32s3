// WS2812 / WS2812B addressable LEDs through the RMT peripheral (GRB, 800 kHz).
#pragma once
#include <stdint.h>
typedef struct ws2812 ws2812;
ws2812 *ws2812_new(int gpio, int count);            // NULL on failure
void ws2812_set(ws2812 *s, int index, uint8_t r, uint8_t g, uint8_t b);
void ws2812_show(ws2812 *s);                        // sends the colours set so far
