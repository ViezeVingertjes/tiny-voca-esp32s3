// ESP32-S3 backend of tv_kernels.h: PIE assembly kernels, matmuls split over both cores.
#include "tv_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

void dot_rows_s8_accx(const int8_t *x, const int8_t *W, int32_t *out, int N, int k16);
void qacc_mm_s8(const int8_t *x, const int8_t *wblk, uint8_t *raw, int k32);

uint64_t tv_time_us(void) { return (uint64_t)esp_timer_get_time(); }

void *tv_alloc(size_t n) {
    void *p = heap_caps_aligned_alloc(16, n + 64, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!p) p = heap_caps_aligned_alloc(16, n + 64, MALLOC_CAP_8BIT);
    if (!p) {
        printf("tv: out of memory (%u bytes, largest PSRAM block %u)\n", (unsigned)n,
               (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
        abort();
    }
    return p;
}

void tv_free(void *p) { heap_caps_free(p); }

// ---------------------------------------------------------------- QACC raw layout
static void pack_qacc(const int32_t *v, uint8_t *raw) {      // 16 values (20-bit) -> 64 bytes
    memset(raw, 0, 64);
    for (int h = 0; h < 2; h++)
        for (int i = 0; i < 8; i++) {
            uint32_t u = (uint32_t)v[8 * h + i] & 0xFFFFF;
            int bit = 20 * i;
            for (int b = 0; b < 20; b++, bit++)
                if (u >> b & 1) raw[32 * h + bit / 8] |= 1 << (bit % 8);
        }
}

static inline void unpack_qacc(const uint8_t *raw, int32_t *o) {
    for (int h = 0; h < 2; h++, raw += 32, o += 8) {
        const uint32_t *w = (const uint32_t *)raw;
        uint32_t w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4];
        o[0] = (int32_t)(w0 << 12) >> 12;
        o[1] = (int32_t)(((w0 >> 20) | (w1 << 12)) << 12) >> 12;
        o[2] = (int32_t)(w1 << 4) >> 12;
        o[3] = (int32_t)(((w1 >> 28) | (w2 << 4)) << 12) >> 12;
        o[4] = (int32_t)(((w2 >> 16) | (w3 << 16)) << 12) >> 12;
        o[5] = (int32_t)(w3 << 8) >> 12;
        o[6] = (int32_t)(((w3 >> 24) | (w4 << 8)) << 12) >> 12;
        o[7] = (int32_t)w4 >> 12;
    }
}

// Backend data per linear: packed preloads [chunks][Npad/16][64 bytes] (or NULL = zeros).
typedef struct {
    uint8_t *raw_pre;
} backend;

int tv_linear_prepare(tv_linear *L) {
    if (L->K % (32 * L->chunks)) return -1;
    if (!L->pre) { L->backend = NULL; return 0; }
    int nblocks = (L->N + 15) / 16;
    backend *b = calloc(1, sizeof *b);
    b->raw_pre = heap_caps_aligned_alloc(16, (size_t)L->chunks * nblocks * 64, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!b->raw_pre) return -1;
    for (int c = 0; c < L->chunks; c++)
        for (int nb = 0; nb < nblocks; nb++) {
            int32_t v[16] = {0};
            for (int j = 0; j < 16 && nb * 16 + j < L->N; j++) v[j] = L->pre[c * L->N + nb * 16 + j];
            pack_qacc(v, b->raw_pre + ((size_t)c * nblocks + nb) * 64);
        }
    L->backend = b;
    return 0;
}

void tv_linear_release(tv_linear *L) {
    backend *b = L->backend;
    if (b) { heap_caps_free(b->raw_pre); free(b); }
    L->backend = NULL;
}

// ---------------------------------------------------------------- matmul over both cores
typedef struct {
    const tv_linear *L;
    const int8_t *x;
    int32_t *acc;
    int T, nb0, nb1;
    long saturations;
} mm_job;

static void mm_run(mm_job *j) {
    static const uint8_t zero[64] __attribute__((aligned(16))) = {0};
    const tv_linear *L = j->L;
    const backend *be = L->backend;
    const int K = L->K, N = L->N, chunk = K / L->chunks, nblocks = (N + 15) / 16;
    uint8_t raw[64] __attribute__((aligned(16)));
    int32_t lanes[16], pre[16], total[16];
    long sat = 0;
    for (int nb = j->nb0; nb < j->nb1; nb++) {
        const int8_t *wb = L->wq + (size_t)nb * K * 16;
        int width = N - nb * 16 < 16 ? N - nb * 16 : 16;
        for (int t = 0; t < j->T; t++) {
            const int8_t *xt = j->x + (size_t)t * K;
            for (int c = 0; c < L->chunks; c++) {
                const uint8_t *init = be ? be->raw_pre + ((size_t)c * nblocks + nb) * 64 : zero;
                memcpy(raw, init, 64);
                qacc_mm_s8(xt + c * chunk, wb + (size_t)c * chunk * 16, raw, chunk / 32);
                unpack_qacc(raw, lanes);
                if (be) unpack_qacc(init, pre); else memset(pre, 0, sizeof pre);
                for (int i = 0; i < width; i++) {
                    if (lanes[i] == TV_QACC_MAX || lanes[i] == -TV_QACC_MAX - 1) sat++;
                    total[i] = (c ? total[i] : 0) + lanes[i] - pre[i];
                }
            }
            memcpy(j->acc + (size_t)t * N + nb * 16, total, width * sizeof(int32_t));
        }
    }
    j->saturations = sat;
}

static SemaphoreHandle_t go_sem, done_sem;
static mm_job *worker_job;

static void worker(void *unused) {
    for (;;) {
        xSemaphoreTake(go_sem, portMAX_DELAY);
        mm_run(worker_job);
        xSemaphoreGive(done_sem);
    }
}

void tv_kernels_init(void) {
    if (go_sem) return;
    go_sem = xSemaphoreCreateBinary();
    done_sem = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCore(worker, "tv_worker", 4096, NULL, configMAX_PRIORITIES - 2, NULL, 1);
}

long tv_matmul_i8(const tv_linear *L, const int8_t *x, int T, int32_t *acc) {
    int nblocks = (L->N + 15) / 16;
    if (nblocks < 2) {
        mm_job j = {L, x, acc, T, 0, nblocks, 0};
        mm_run(&j);
        return j.saturations;
    }
    int split = nblocks / 2;
    mm_job a = {L, x, acc, T, 0, split, 0}, b = {L, x, acc, T, split, nblocks, 0};
    worker_job = &b;
    xSemaphoreGive(go_sem);
    mm_run(&a);
    xSemaphoreTake(done_sem, portMAX_DELAY);
    return a.saturations + b.saturations;
}

void tv_dot_rows_i8(const int8_t *x, const int8_t *W, int N, int K, int32_t *out) {
    dot_rows_s8_accx(x, W, out, N, K / 16);
}
