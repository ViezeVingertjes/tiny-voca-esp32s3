// Speaker output for the Waveshare ESP32-S3-AUDIO-Board: ES8311 DAC over I2S, NS4150B amplifier
// enabled through pin 8 of the TCA9555 I/O expander. Pins from the Waveshare wiki:
//   I2C SDA 11, SCL 10; I2S MCLK 12, BCLK 13, WS 14, DOUT 16.
// Only expander port 1 bit 0 (PA enable) is ever written: port 0 holds the USB/camera mux.
#include "audio.h"
#include <stdio.h>
#include <string.h>
#include "driver/i2c_master.h"
#include "driver/i2s_std.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "tv_kernels.h"

#define PIN_SDA 11
#define PIN_SCL 10
#define PIN_MCLK 12
#define PIN_BCLK 13
#define PIN_WS 14
#define PIN_DOUT 16
#define ES8311_ADDR 0x18
#define TCA9555_ADDR 0x20
#define SAMPLE_RATE 24000

static i2c_master_bus_handle_t bus;
static i2c_master_dev_handle_t codec, expander;
static i2s_chan_handle_t tx;
static QueueHandle_t queue;

static esp_err_t wr(i2c_master_dev_handle_t dev, uint8_t reg, uint8_t value) {
    uint8_t buf[2] = {reg, value};
    return i2c_master_transmit(dev, buf, 2, 100);
}

static esp_err_t rd(i2c_master_dev_handle_t dev, uint8_t reg, uint8_t *value) {
    return i2c_master_transmit_receive(dev, &reg, 1, value, 1, 100);
}

static int amp_enable(void) {
    uint8_t out, cfg;
    if (rd(expander, 0x03, &out) || rd(expander, 0x07, &cfg)) return -1;    // port 1 output / config
    if (wr(expander, 0x03, out | 0x01)) return -1;                          // P1.0 high
    if (wr(expander, 0x07, cfg & ~0x01)) return -1;                         // P1.0 output, others untouched
    return 0;
}

// ES8311: slave, MCLK from pin = 256 * fs, 16-bit I2S (register values from Espressif's es8311 driver).
static int codec_init(int volume) {
    uint8_t id1 = 0, id2 = 0, v;
    if (rd(codec, 0xFD, &id1) || rd(codec, 0xFE, &id2)) return -1;
    printf("audio: ES8311 chip id %02x%02x\n", id1, id2);
    if (id1 != 0x83 || id2 != 0x11) return -2;
    wr(codec, 0x00, 0x1F);
    vTaskDelay(pdMS_TO_TICKS(20));
    wr(codec, 0x00, 0x00);
    wr(codec, 0x00, 0x80);                                  // power on
    wr(codec, 0x01, 0x3F);                                  // all clocks on, MCLK from pin
    rd(codec, 0x06, &v); wr(codec, 0x06, v & ~0x20);        // BCLK not inverted
    // 6.144 MHz / 24 kHz: pre_div 1, mult 1x, adc/dac div 1, single speed, lrck 0x00ff, osr 0x10
    rd(codec, 0x02, &v); wr(codec, 0x02, (v & 0x07) | (0 << 5) | (0 << 3));
    wr(codec, 0x03, 0x10);
    wr(codec, 0x04, 0x10);
    wr(codec, 0x05, 0x00);
    rd(codec, 0x06, &v); wr(codec, 0x06, (v & 0xE0) | (4 - 1));
    rd(codec, 0x07, &v); wr(codec, 0x07, (v & 0xC0) | 0x00);
    wr(codec, 0x08, 0xFF);
    rd(codec, 0x00, &v); wr(codec, 0x00, v & 0xBF);        // slave serial port
    wr(codec, 0x09, 0x0C);                                  // SDP in: I2S, 16-bit
    wr(codec, 0x0A, 0x0C);                                  // SDP out: I2S, 16-bit
    wr(codec, 0x0D, 0x01);                                  // analog power up
    wr(codec, 0x0E, 0x02);
    wr(codec, 0x12, 0x00);                                  // DAC power up
    wr(codec, 0x13, 0x10);                                  // output to HP drive
    wr(codec, 0x1C, 0x6A);
    wr(codec, 0x37, 0x08);                                  // bypass DAC equalizer
    wr(codec, 0x32, volume <= 0 ? 0 : volume * 256 / 100 - 1);
    rd(codec, 0x31, &v); wr(codec, 0x31, v & ~0x60);        // unmute
    return 0;
}

typedef struct {
    int16_t *pcm;
    int n;
} chunk;

static void player(void *arg) {
    chunk c;
    for (;;) {
        if (xQueueReceive(queue, &c, portMAX_DELAY) != pdTRUE) continue;
        if (!c.pcm) continue;
        size_t written;
        i2s_channel_write(tx, c.pcm, (size_t)c.n * 2, &written, portMAX_DELAY);
        tv_free(c.pcm);
    }
}

int audio_init(int volume) {
    i2c_master_bus_config_t bc = {.i2c_port = 0, .sda_io_num = PIN_SDA, .scl_io_num = PIN_SCL,
                                  .clk_source = I2C_CLK_SRC_DEFAULT, .glitch_ignore_cnt = 7,
                                  .flags.enable_internal_pullup = true};
    if (i2c_new_master_bus(&bc, &bus)) return -1;
    i2c_device_config_t dc = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = ES8311_ADDR, .scl_speed_hz = 100000};
    i2c_master_bus_add_device(bus, &dc, &codec);
    dc.device_address = TCA9555_ADDR;
    i2c_master_bus_add_device(bus, &dc, &expander);

    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    cc.dma_frame_num = 480;
    cc.auto_clear = true;                                   // silence when starved
    if (i2s_new_channel(&cc, &tx, NULL)) return -2;
    i2s_std_config_t sc = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {.mclk = PIN_MCLK, .bclk = PIN_BCLK, .ws = PIN_WS, .dout = PIN_DOUT, .din = I2S_GPIO_UNUSED},
    };
    sc.clk_cfg.mclk_multiple = I2S_MCLK_MULTIPLE_256;
    if (i2s_channel_init_std_mode(tx, &sc) || i2s_channel_enable(tx)) return -3;
    int rc = codec_init(volume);                            // MCLK must run before the codec is set up
    if (rc) { printf("audio: ES8311 init failed (%d)\n", rc); return -4; }
    if (amp_enable()) { printf("audio: TCA9555 not answering, amplifier left off\n"); return -5; }
    queue = xQueueCreate(8, sizeof(chunk));
    xTaskCreatePinnedToCore(player, "player", 4096, NULL, 10, NULL, 0);
    printf("audio: ready (24 kHz, volume %d)\n", volume);
    return 0;
}

void audio_play(int16_t *pcm, int n) {
    chunk c = {pcm, n};
    if (!queue) { tv_free(pcm); return; }
    xQueueSend(queue, &c, portMAX_DELAY);
}

int audio_pending(void) { return queue ? (int)uxQueueMessagesWaiting(queue) : 0; }

void audio_beep(int hz, int ms) {
    int n = SAMPLE_RATE * ms / 1000;
    int16_t *pcm = tv_alloc((size_t)n * 2);
    for (int i = 0; i < n; i++) {
        float env = i < 240 ? i / 240.0f : (n - i < 240 ? (n - i) / 240.0f : 1.0f);
        float phase = (float)((long long)i * hz % SAMPLE_RATE) / SAMPLE_RATE;
        float s = phase < 0.5f ? 4 * phase - 1 : 3 - 4 * phase;            // triangle
        pcm[i] = (int16_t)(s * env * 8000);
    }
    audio_play(pcm, n);
}
