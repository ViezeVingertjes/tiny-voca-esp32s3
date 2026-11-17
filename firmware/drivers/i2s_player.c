#include "i2s_player.h"
#include "driver/i2s_std.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "tv_kernels.h"

static i2s_chan_handle_t tx;
static QueueHandle_t queue;
static int rate;

typedef struct {
    int16_t *pcm;
    int n;
} chunk;

static void player(void *arg) {
    chunk c;
    for (;;) {
        if (xQueueReceive(queue, &c, portMAX_DELAY) != pdTRUE || !c.pcm) continue;
        size_t written;
        i2s_channel_write(tx, c.pcm, (size_t)c.n * 2, &written, portMAX_DELAY);
        tv_free(c.pcm);
    }
}

int i2s_player_start(const i2s_player_pins *p) {
    rate = p->sample_rate;
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    cc.dma_frame_num = 480;
    cc.auto_clear = true;                                   // silence when starved
    if (i2s_new_channel(&cc, &tx, NULL)) return -1;
    i2s_std_config_t sc = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(p->sample_rate),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {.mclk = p->mclk, .bclk = p->bclk, .ws = p->ws, .dout = p->dout, .din = I2S_GPIO_UNUSED},
    };
    sc.clk_cfg.mclk_multiple = I2S_MCLK_MULTIPLE_256;
    if (i2s_channel_init_std_mode(tx, &sc) || i2s_channel_enable(tx)) return -2;
    queue = xQueueCreate(8, sizeof(chunk));
    xTaskCreatePinnedToCore(player, "player", 4096, NULL, 10, NULL, 0);
    return 0;
}

void i2s_player_play(int16_t *pcm, int n) {
    chunk c = {pcm, n};
    if (!queue) { tv_free(pcm); return; }
    xQueueSend(queue, &c, portMAX_DELAY);
}

int i2s_player_pending(void) { return queue ? (int)uxQueueMessagesWaiting(queue) : 0; }

void i2s_player_beep(int hz, int ms) {
    int n = rate * ms / 1000;
    int16_t *pcm = tv_alloc((size_t)n * 2);
    for (int i = 0; i < n; i++) {
        float env = i < 240 ? i / 240.0f : (n - i < 240 ? (n - i) / 240.0f : 1.0f);
        float phase = (float)((long long)i * hz % rate) / rate;
        float s = phase < 0.5f ? 4 * phase - 1 : 3 - 4 * phase;            // triangle
        pcm[i] = (int16_t)(s * env * 8000);
    }
    i2s_player_play(pcm, n);
}
