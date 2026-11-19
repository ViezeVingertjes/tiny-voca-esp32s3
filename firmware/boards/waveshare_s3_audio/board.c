// Waveshare ESP32-S3-AUDIO-Board: ES8311 DAC over I2S, NS4150B amplifier enabled through pin 8 of
// the TCA9555 I/O expander, and a ring of 7 WS2812B LEDs used as a sound level meter.
// Pins (Waveshare wiki, community pin map): I2C SDA 11, SCL 10; I2S MCLK 12, BCLK 13, WS 14,
// DOUT 16; LEDs 38.
// Only expander port 1 bit 0 (PA enable) is ever written: port 0 holds the USB/camera mux, and
// driving pins 6/7 the wrong way kills USB.
#include "board.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "driver/i2c_master.h"
#include "es8311.h"
#include "i2c_reg.h"
#include "i2s_player.h"
#include "ws2812.h"

#define TCA9555_ADDR 0x20

static i2c_master_bus_handle_t bus;
static i2c_master_dev_handle_t codec, expander;
static ws2812 *leds;

// ---------------------------------------------------------------- LED level meter
#define LED_PIN 38
#define LED_COUNT 7
#define LED_BRIGHTNESS 40.0f        // of 255: the LEDs are very bright
#define METER_FLOOR_DB (-45.0f)     // no LED lit below this
#define METER_TOP_DB (-9.0f)        // all LEDs lit above this
#define METER_FALL 0.3f             // LEDs per 10 ms update while the level drops

// Called every 10 ms by the player with the RMS (0..1) of what is being heard.
static void meter(float rms) {
    static float shown;
    static uint8_t last[LED_COUNT][3];
    float db = rms > 1e-6f ? 20.0f * log10f(rms) : -120.0f;
    float target = (db - METER_FLOOR_DB) / (METER_TOP_DB - METER_FLOOR_DB) * LED_COUNT;
    target = target < 0 ? 0 : target > LED_COUNT ? LED_COUNT : target;
    shown = target > shown ? target : (shown - METER_FALL > target ? shown - METER_FALL : target);
    uint8_t now[LED_COUNT][3];
    for (int i = 0; i < LED_COUNT; i++) {
        float on = shown - i;
        on = on < 0 ? 0 : on > 1 ? 1 : on;
        on = on * on * LED_BRIGHTNESS;                      // rough gamma for the partly lit LED
        float r = i < 4 ? 0.0f : 1.0f, g = i < 4 ? 1.0f : i < 6 ? 0.7f : 0.0f;   // green, yellow, red
        now[i][0] = (uint8_t)(r * on); now[i][1] = (uint8_t)(g * on); now[i][2] = 0;
    }
    if (!memcmp(now, last, sizeof now)) return;            // nothing changed: leave the LEDs alone
    memcpy(last, now, sizeof now);
    for (int i = 0; i < LED_COUNT; i++) ws2812_set(leds, i, now[i][0], now[i][1], now[i][2]);
    ws2812_show(leds);
}

const char *board_name(void) { return "waveshare_s3_audio"; }

static int amp_enable(void) {
    uint8_t out, cfg;
    if (i2c_reg_read(expander, 0x03, &out) || i2c_reg_read(expander, 0x07, &cfg)) return -1;   // port 1 output / config
    if (i2c_reg_write(expander, 0x03, out | 0x01)) return -1;                                 // P1.0 high
    if (i2c_reg_write(expander, 0x07, cfg & ~0x01)) return -1;                                // P1.0 output, others untouched
    return 0;
}

int board_init(int volume) {
    i2s_player_pins pins = {.mclk = 12, .bclk = 13, .ws = 14, .dout = 16, .sample_rate = 24000};
    int rc = es8311_audio_start(11, 10, &pins, volume, &bus, &codec);
    if (rc == -1 || rc == -2) return rc;
    leds = ws2812_new(LED_PIN, LED_COUNT);
    if (leds) {
        ws2812_show(leds);                                  // all off
        i2s_player_set_meter(meter);
    } else {
        printf("leds: RMT setup failed, no level meter\n");
    }
    if (rc) return rc;
    i2c_device_config_t dc = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = TCA9555_ADDR, .scl_speed_hz = 100000};
    if (i2c_master_bus_add_device(bus, &dc, &expander) || amp_enable()) {
        printf("audio: TCA9555 not answering, amplifier left off\n");
        return -4;
    }
    printf("audio: ready (24 kHz, volume %d)\n", volume);
    return 0;
}

void board_play(int16_t *pcm, int n) { i2s_player_play(pcm, n); }
int board_set_volume(int volume) { return codec ? es8311_set_volume(codec, volume) : -1; }
void board_beep(int hz, int ms) { i2s_player_beep(hz, ms); }
void board_show_text(const char *text) { (void)text; }   // no screen
