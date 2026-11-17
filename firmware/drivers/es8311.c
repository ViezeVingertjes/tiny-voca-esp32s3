// ES8311 setup for playback (register values from Espressif's es8311 driver in esp-bsp).
#include "es8311.h"
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static esp_err_t wr(i2c_master_dev_handle_t dev, uint8_t reg, uint8_t value) {
    uint8_t buf[2] = {reg, value};
    return i2c_master_transmit(dev, buf, 2, 100);
}

static esp_err_t rd(i2c_master_dev_handle_t dev, uint8_t reg, uint8_t *value) {
    return i2c_master_transmit_receive(dev, &reg, 1, value, 1, 100);
}

int es8311_set_volume(i2c_master_dev_handle_t dev, int volume) {
    if (volume < 0) volume = 0;
    if (volume > 100) volume = 100;
    return wr(dev, 0x32, volume == 0 ? 0 : volume * 256 / 100 - 1) ? -1 : 0;
}

int es8311_start(i2c_master_dev_handle_t dev, int volume) {
    uint8_t id1 = 0, id2 = 0, v;
    if (rd(dev, 0xFD, &id1) || rd(dev, 0xFE, &id2)) return -1;
    printf("audio: ES8311 chip id %02x%02x\n", id1, id2);
    if (id1 != 0x83 || id2 != 0x11) return -2;
    wr(dev, 0x00, 0x1F);
    vTaskDelay(pdMS_TO_TICKS(20));
    wr(dev, 0x00, 0x00);
    wr(dev, 0x00, 0x80);                                  // power on
    wr(dev, 0x01, 0x3F);                                  // all clocks on, MCLK from pin
    rd(dev, 0x06, &v); wr(dev, 0x06, v & ~0x20);          // BCLK not inverted
    // 6.144 MHz / 24 kHz: pre_div 1, mult 1x, adc/dac div 1, single speed, lrck 0x00ff, osr 0x10
    rd(dev, 0x02, &v); wr(dev, 0x02, (v & 0x07) | (0 << 5) | (0 << 3));
    wr(dev, 0x03, 0x10);
    wr(dev, 0x04, 0x10);
    wr(dev, 0x05, 0x00);
    rd(dev, 0x06, &v); wr(dev, 0x06, (v & 0xE0) | (4 - 1));
    rd(dev, 0x07, &v); wr(dev, 0x07, (v & 0xC0) | 0x00);
    wr(dev, 0x08, 0xFF);
    rd(dev, 0x00, &v); wr(dev, 0x00, v & 0xBF);          // slave serial port
    wr(dev, 0x09, 0x0C);                                  // SDP in: I2S, 16-bit
    wr(dev, 0x0A, 0x0C);                                  // SDP out: I2S, 16-bit
    wr(dev, 0x0D, 0x01);                                  // analog power up
    wr(dev, 0x0E, 0x02);
    wr(dev, 0x12, 0x00);                                  // DAC power up
    wr(dev, 0x13, 0x10);                                  // output to HP drive
    wr(dev, 0x1C, 0x6A);
    wr(dev, 0x37, 0x08);                                  // bypass DAC equalizer
    es8311_set_volume(dev, volume);
    rd(dev, 0x31, &v); wr(dev, 0x31, v & ~0x60);          // unmute
    return 0;
}
