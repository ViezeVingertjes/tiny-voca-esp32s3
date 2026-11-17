// Waveshare ESP32-S3-AUDIO-Board: ES8311 DAC over I2S, NS4150B amplifier enabled through pin 8 of
// the TCA9555 I/O expander. Pins from the Waveshare wiki:
//   I2C SDA 11, SCL 10; I2S MCLK 12, BCLK 13, WS 14, DOUT 16.
// Only expander port 1 bit 0 (PA enable) is ever written: port 0 holds the USB/camera mux, and
// driving pins 6/7 the wrong way kills USB.
#include "board.h"
#include <stdio.h>
#include "driver/i2c_master.h"
#include "es8311.h"
#include "i2s_player.h"

#define TCA9555_ADDR 0x20
#define ES8311_ADDR 0x18

static i2c_master_bus_handle_t bus;
static i2c_master_dev_handle_t codec, expander;

const char *board_name(void) { return "waveshare_s3_audio"; }

static esp_err_t rd(i2c_master_dev_handle_t dev, uint8_t reg, uint8_t *value) {
    return i2c_master_transmit_receive(dev, &reg, 1, value, 1, 100);
}

static esp_err_t wr(i2c_master_dev_handle_t dev, uint8_t reg, uint8_t value) {
    uint8_t buf[2] = {reg, value};
    return i2c_master_transmit(dev, buf, 2, 100);
}

static int amp_enable(void) {
    uint8_t out, cfg;
    if (rd(expander, 0x03, &out) || rd(expander, 0x07, &cfg)) return -1;    // port 1 output / config
    if (wr(expander, 0x03, out | 0x01)) return -1;                          // P1.0 high
    if (wr(expander, 0x07, cfg & ~0x01)) return -1;                         // P1.0 output, others untouched
    return 0;
}

int board_init(int volume) {
    i2c_master_bus_config_t bc = {.i2c_port = 0, .sda_io_num = 11, .scl_io_num = 10, .clk_source = I2C_CLK_SRC_DEFAULT,
                                  .glitch_ignore_cnt = 7, .flags.enable_internal_pullup = true};
    if (i2c_new_master_bus(&bc, &bus)) return -1;
    i2c_device_config_t dc = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = ES8311_ADDR, .scl_speed_hz = 100000};
    i2c_master_bus_add_device(bus, &dc, &codec);
    dc.device_address = TCA9555_ADDR;
    i2c_master_bus_add_device(bus, &dc, &expander);
    i2s_player_pins pins = {.mclk = 12, .bclk = 13, .ws = 14, .dout = 16, .sample_rate = 24000};
    if (i2s_player_start(&pins)) return -2;                 // MCLK must run before the codec is set up
    int rc = es8311_start(codec, volume);
    if (rc) { printf("audio: ES8311 init failed (%d)\n", rc); codec = NULL; return -3; }
    if (amp_enable()) { printf("audio: TCA9555 not answering, amplifier left off\n"); return -4; }
    printf("audio: ready (24 kHz, volume %d)\n", volume);
    return 0;
}

void board_play(int16_t *pcm, int n) { i2s_player_play(pcm, n); }
int board_set_volume(int volume) { return codec ? es8311_set_volume(codec, volume) : -1; }
void board_beep(int hz, int ms) { i2s_player_beep(hz, ms); }
void board_show_text(const char *text) { (void)text; }   // no screen
