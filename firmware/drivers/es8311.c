// ES8311 setup for playback (register values from Espressif's es8311 driver in esp-bsp).
#include "es8311.h"
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "i2c_reg.h"

int es8311_set_volume(i2c_master_dev_handle_t dev, int volume) {
    if (volume < 0) volume = 0;
    if (volume > 100) volume = 100;
    return i2c_reg_write(dev, 0x32, volume == 0 ? 0 : volume * 256 / 100 - 1) ? -1 : 0;
}

int es8311_start(i2c_master_dev_handle_t dev, int volume) {
    uint8_t id1 = 0, id2 = 0;
    if (i2c_reg_read(dev, 0xFD, &id1) || i2c_reg_read(dev, 0xFE, &id2)) return -1;
    printf("audio: ES8311 chip id %02x%02x\n", id1, id2);
    if (id1 != 0x83 || id2 != 0x11) return -2;
    esp_err_t err = i2c_reg_write(dev, 0x00, 0x1F);
    vTaskDelay(pdMS_TO_TICKS(20));
    err |= i2c_reg_write(dev, 0x00, 0x00);
    err |= i2c_reg_write(dev, 0x00, 0x80);                // power on
    err |= i2c_reg_write(dev, 0x01, 0x3F);                // all clocks on, MCLK from pin
    err |= i2c_reg_update(dev, 0x06, 0xDF, 0x00);         // BCLK not inverted
    // 6.144 MHz / 24 kHz: pre_div 1, mult 1x, adc/dac div 1, single speed, lrck 0x00ff, osr 0x10
    err |= i2c_reg_update(dev, 0x02, 0x07, 0x00);
    err |= i2c_reg_write(dev, 0x03, 0x10);
    err |= i2c_reg_write(dev, 0x04, 0x10);
    err |= i2c_reg_write(dev, 0x05, 0x00);
    err |= i2c_reg_update(dev, 0x06, 0xE0, 4 - 1);
    err |= i2c_reg_update(dev, 0x07, 0xC0, 0x00);
    err |= i2c_reg_write(dev, 0x08, 0xFF);
    err |= i2c_reg_update(dev, 0x00, 0xBF, 0x00);         // slave serial port
    err |= i2c_reg_write(dev, 0x09, 0x0C);                // SDP in: I2S, 16-bit
    err |= i2c_reg_write(dev, 0x0A, 0x0C);                // SDP out: I2S, 16-bit
    err |= i2c_reg_write(dev, 0x0D, 0x01);                // analog power up
    err |= i2c_reg_write(dev, 0x0E, 0x02);
    err |= i2c_reg_write(dev, 0x12, 0x00);                // DAC power up
    err |= i2c_reg_write(dev, 0x13, 0x10);                // output to HP drive
    err |= i2c_reg_write(dev, 0x1C, 0x6A);
    err |= i2c_reg_write(dev, 0x37, 0x08);                // bypass DAC equalizer
    err |= es8311_set_volume(dev, volume) ? ESP_FAIL : ESP_OK;
    err |= i2c_reg_update(dev, 0x31, 0x9F, 0x00);         // unmute
    return err ? -3 : 0;
}

int es8311_audio_start(int sda, int scl, const i2s_player_pins *i2s, int volume, i2c_master_bus_handle_t *bus,
                       i2c_master_dev_handle_t *codec) {
    *codec = NULL;
    i2c_master_bus_config_t bc = {.i2c_port = 0, .sda_io_num = sda, .scl_io_num = scl, .clk_source = I2C_CLK_SRC_DEFAULT,
                                  .glitch_ignore_cnt = 7, .flags.enable_internal_pullup = true};
    if (i2c_new_master_bus(&bc, bus)) return -1;
    i2c_device_config_t dc = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = ES8311_ADDR, .scl_speed_hz = 100000};
    i2c_master_dev_handle_t dev;
    if (i2c_master_bus_add_device(*bus, &dc, &dev)) return -1;
    if (i2s_player_start(i2s)) return -2;
    int rc = es8311_start(dev, volume);
    if (rc) { printf("audio: ES8311 init failed (%d)\n", rc); return -3; }
    *codec = dev;
    return 0;
}
