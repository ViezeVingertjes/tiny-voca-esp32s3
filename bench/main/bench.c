// tiny-voca ESP32-S3 microbenchmarks: int8 matmul kernels, fp32 baseline,
// memory bandwidth and the scalar math the vocoder needs.
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <inttypes.h>
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

int32_t dot_rows_s8_accx(const int8_t *x, const int8_t *W, int32_t *out, int N, int k16);
void dot_cols_s8_accx(const int8_t *X, const int8_t *w, int32_t *out, int T, int k16, int out_stride_bytes);
void qacc_block_s8(const int8_t *x, const int8_t *wblk, uint8_t *qacc_out, int k16);
void read_vld(const void *p, int n16);
void qacc_block_s8_v2(const int8_t *x, const int8_t *wblk, uint8_t *qacc_out, int k32);
void qacc_probe(const int8_t *w, const int8_t *x, int n, uint8_t *out);
void qacc_block_s8_q8(const int8_t *x, const int8_t *wblk, int8_t *out, int k32, const int32_t *mult, const uint8_t *bias_raw);

#define MHZ 240
#define SRAM (MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
#define PSRAM (MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)

static void *alloc(size_t n, uint32_t caps) {
    void *p = heap_caps_aligned_alloc(16, n + 16, caps);   // +16: kernels over-read
    if (!p) {
        printf("ALLOC FAIL %u bytes caps=%" PRIx32 "\n", (unsigned)n, caps);
        abort();
    }
    memset(p, 0, n + 16);
    return p;
}

static uint32_t rng = 12345;
static int8_t rand_s8(int range) {   // uniform in [-range, range-1]
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (int8_t)((int)(rng % (2 * range)) - range);
}

static inline uint32_t now(void) { return esp_cpu_get_cycle_count(); }

static void report(const char *name, uint32_t cycles, double macs) {
    printf("  %-44s %9" PRIu32 " cyc  %6.2f MAC/cyc  %7.1f MMAC/s\n",
           name, cycles, macs / cycles, macs / cycles * MHZ);
}

static int check(const char *name, const int32_t *got, const int32_t *ref, int n) {
    for (int i = 0; i < n; i++) {
        if (got[i] != ref[i]) {
            printf("  !! %s mismatch at %d: got %" PRId32 " want %" PRId32 "\n", name, i, got[i], ref[i]);
            return 0;
        }
    }
    return 1;
}

// Sign-extend lane i (20 bits) of the 160-bit accumulator half at p.
static inline int32_t lane20(const uint8_t *p, int i) {
    int bit = 20 * i, byte = bit >> 3, shift = bit & 7;
    uint32_t v = p[byte] | (p[byte + 1] << 8) | (p[byte + 2] << 16) | ((uint32_t)p[byte + 3] << 24);
    return ((int32_t)((v >> shift) << 12)) >> 12;
}

static void unpack_qacc(const uint8_t *raw, int32_t *out) {
    uint8_t l[24], h[24];
    memcpy(l, raw, 16); memcpy(l + 16, raw + 16, 4); memset(l + 20, 0, 4);
    memcpy(h, raw + 32, 16); memcpy(h + 16, raw + 48, 4); memset(h + 20, 0, 4);
    for (int i = 0; i < 8; i++) {
        out[i] = lane20(l, i);
        out[8 + i] = lane20(h, i);
    }
}

// ---------------------------------------------------------------- QACC probe
static void print_raw_lanes(const char *label, const uint8_t *raw) {
    int32_t v[16];
    unpack_qacc(raw, v);
    printf("    %-10s", label);
    for (int i = 0; i < 16; i++) printf(" %7" PRId32, v[i]);
    printf("\n");
}

static void probe_case(const char *name, const int8_t *w, const int8_t *x, int n) {
    uint8_t buf[256] __attribute__((aligned(16)));
    memset(buf, 0xAA, sizeof buf);
    qacc_probe(w, x, n, buf);
    printf("  %s (n=%d), expected lane0 = %d\n", name, n, w[0] * x[0] * n);
    print_raw_lanes("raw", buf);
    printf("    %-10s", "srcmb.s16");
    for (int i = 0; i < 8; i++) printf(" %7d", ((int16_t *)(buf + 64))[i]);
    printf("   (8 x int16)\n");
    print_raw_lanes("raw after", buf + 80);
    printf("    %-10s", "srcmb.s8>>8");
    for (int i = 0; i < 16; i++) printf(" %7d", ((int8_t *)(buf + 144))[i]);
    printf("\n");
    print_raw_lanes("raw after", buf + 160);
}

static void bench_probe(void) {
    printf("\n== QACC semantics probe ==\n");
    int8_t w[16] __attribute__((aligned(16))), x[16] __attribute__((aligned(16)));
    for (int i = 0; i < 16; i++) { w[i] = (int8_t)(i - 8); x[i] = (int8_t)(3 * i + 1); }
    probe_case("small mixed", w, x, 100);
    for (int i = 0; i < 16; i++) { w[i] = 127; x[i] = 127; }
    probe_case("positive overflow", w, x, 40);
    for (int i = 0; i < 16; i++) { w[i] = 127; x[i] = -128; }
    probe_case("negative overflow", w, x, 40);
}

// ---------------------------------------------------------------- int8 matmul
// Fast unpack of the 2 x 160-bit raw QACC dump into 16 int32 lanes.
static inline void unpack_fast(const uint8_t *raw, int32_t *o) {
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

// Inverse of unpack_fast: 16 int32 values (must fit 20 bits) -> raw QACC layout.
static void pack_qacc(const int32_t *v, uint8_t *raw) {
    memset(raw, 0, 64);
    for (int h = 0; h < 2; h++)
        for (int i = 0; i < 8; i++) {
            uint32_t u = (uint32_t)v[8 * h + i] & 0xFFFFF;
            int bit = 20 * i;
            for (int b = 0; b < 20; b++, bit++)
                if (u >> b & 1) raw[32 * h + bit / 8] |= 1 << (bit % 8);
        }
    // bytes 16..19 of each half hold bits 128..159: already in place (raw[16..19], raw[48..51])
}

static inline int8_t requant_hi(int32_t acc, int32_t m) {
    int32_t q = (int32_t)(((int64_t)acc * m) >> 32);
    return q > 127 ? 127 : q < -128 ? -128 : q;
}

typedef struct {
    const int8_t *X, *Wp;
    int32_t *out;          // mode 1: int32 results [T][N]
    int8_t *out8;          // mode 2: requantized int8 results [T][N]
    const int32_t *mult;   // mode 2: per-channel multiplier, q = (acc * mult + 2^15) >> 16
    const int32_t *mhi;    // mode 3: fused kernel, q = (acc+bias) * mhi >> 32
    const uint8_t *bias_raw;
    int T, N, K, nb0, nb1, mode;
} gemm_job_t;

static inline int8_t requant(int32_t acc, int32_t m) {
    int32_t q = (acc * m + (1 << 15)) >> 16;
    return q > 127 ? 127 : q < -128 ? -128 : q;
}

static void gemm_v2(void *arg) {
    gemm_job_t *j = arg;
    uint8_t raw[64] __attribute__((aligned(16)));
    int32_t acc[16];
    for (int nb = j->nb0; nb < j->nb1; nb++) {
        const int8_t *wb = j->Wp + nb * j->K * 16;
        const int32_t *m = j->mult + nb * 16;
        for (int t = 0; t < j->T; t++) {
            if (j->mode != 3) qacc_block_s8_v2(j->X + t * j->K, wb, raw, j->K / 32);
            if (j->mode == 1) {
                unpack_fast(raw, j->out + t * j->N + nb * 16);
            } else if (j->mode == 3) {
                qacc_block_s8_q8(j->X + t * j->K, wb, j->out8 + t * j->N + nb * 16, j->K / 32,
                                 j->mhi + nb * 16, j->bias_raw + nb * 64);
                continue;
            } else if (j->mode == 2) {
                unpack_fast(raw, acc);
                int8_t *o = j->out8 + t * j->N + nb * 16;
                for (int i = 0; i < 16; i++) o[i] = requant(acc[i], m[i]);
            }
        }
    }
}

// Second core: runs one job when signalled.
static SemaphoreHandle_t go_sem, done_sem;
static void (*job_fn)(void *);
static void *job_arg;
static void worker(void *unused) {
    for (;;) {
        xSemaphoreTake(go_sem, portMAX_DELAY);
        job_fn(job_arg);
        xSemaphoreGive(done_sem);
    }
}
static void run_both(void (*fn)(void *), void *a0, void *a1) {
    job_fn = fn; job_arg = a1;
    xSemaphoreGive(go_sem);
    fn(a0);
    xSemaphoreTake(done_sem, portMAX_DELAY);
}

static uint32_t time_gemm(gemm_job_t *full, int cores) {
    gemm_job_t h0 = *full, h1 = *full;
    h0.nb1 = h1.nb0 = full->N / 32;
    uint32_t c = 0;
    for (int rep = 0; rep < 2; rep++) {      // rep 0 warms caches
        uint32_t t0 = now();
        if (cores == 1) gemm_v2(full); else run_both(gemm_v2, &h0, &h1);
        c = now() - t0;
    }
    return c;
}

static void bench_gemm_s8(void) {
    const int T = 64, K = 192, N = 640;
    printf("\n== int8 matmul: out[T=%d][N=%d] = X[T][K=%d] . W[N][K] ==\n", T, N, K);
    int8_t *X = alloc(T * K, SRAM);
    int8_t *W = alloc(N * K, PSRAM);
    int8_t *Wp_ps = alloc(N * K + 48, PSRAM);
    int32_t *ref = alloc(T * N * 4, PSRAM);
    int8_t *ref8 = alloc(T * N, PSRAM);
    int32_t *mult = alloc(N * 4, SRAM);
    int8_t *out8 = alloc(T * N, SRAM);
    int32_t *mhi = alloc(N * 4, SRAM), *bias = alloc(N * 4, SRAM);
    uint8_t *bias_raw = alloc(N / 16 * 64, SRAM);
    int8_t *ref83 = alloc(T * N, PSRAM);
    for (int n = 0; n < N; n++) { mhi[n] = 6000000 + 10000 * n; bias[n] = (int32_t)(rand_s8(64)) * 300; }
    for (int nb = 0; nb < N / 16; nb++) pack_qacc(bias + nb * 16, bias_raw + nb * 64);
    for (int i = 0; i < T * K; i++) X[i] = rand_s8(64);
    for (int i = 0; i < N * K; i++) W[i] = rand_s8(64);
    for (int n = 0; n < N; n++) mult[n] = 100 + n % 80;
    for (int nb = 0; nb < N / 16; nb++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < 16; j++)
                Wp_ps[(nb * K + k) * 16 + j] = W[(nb * 16 + j) * K + k];
    for (int t = 0; t < T; t++)
        for (int n = 0; n < N; n++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++) s += X[t * K + k] * W[n * K + k];
            ref[t * N + n] = s;
            ref8[t * N + n] = requant(s, mult[n]);
            ref83[t * N + n] = requant_hi(s + bias[n], mhi[n]);
        }
    const double macs = (double)T * N * K;
    char name[80];
    uint32_t c;

    // Weights in SRAM: kernel limit and int8 output.
    int8_t *Wp_sr = alloc(N * K + 48, SRAM);
    memcpy(Wp_sr, Wp_ps, N * K);
    gemm_job_t job = {X, Wp_sr, NULL, out8, mult, mhi, bias_raw, T, N, K, 0, N / 16, 0};
    for (int cores = 1; cores <= 2; cores++) {
        job.mode = 0; c = time_gemm(&job, cores);
        snprintf(name, sizeof name, "W SRAM, kernel only, %d core", cores); report(name, c, macs);
        job.mode = 2; memset(out8, 0, T * N); c = time_gemm(&job, cores);
        snprintf(name, sizeof name, "W SRAM, -> int8 SRAM, %d core", cores); report(name, c, macs);
        if (memcmp(out8, ref8, T * N)) printf("  !! int8 output mismatch\n");
        job.mode = 3; memset(out8, 0, T * N); c = time_gemm(&job, cores);
        snprintf(name, sizeof name, "W %s, fused bias+int8 asm, %d core", "SRAM", cores); report(name, c, macs);
        if (memcmp(out8, ref83, T * N)) {
            int i = 0; while (out8[i] == ref83[i]) i++;
            printf("  !! fused int8 mismatch at %d (t=%d n=%d): got %d want %d\n", i, i / N, i % N, out8[i], ref83[i]);
        }
    }
    heap_caps_free(Wp_sr);

    // Weights in PSRAM (the real engine), outputs in SRAM.
    int32_t *out32 = alloc(T * N * 4, SRAM);
    job.Wp = Wp_ps; job.out = out32;
    for (int cores = 1; cores <= 2; cores++) {
        job.mode = 0; c = time_gemm(&job, cores);
        snprintf(name, sizeof name, "W PSRAM, kernel only, %d core", cores); report(name, c, macs);
        job.mode = 1; memset(out32, 0, T * N * 4); c = time_gemm(&job, cores);
        snprintf(name, sizeof name, "W PSRAM, -> int32 SRAM, %d core", cores); report(name, c, macs);
        check(name, out32, ref, T * N);
        job.mode = 2; memset(out8, 0, T * N); c = time_gemm(&job, cores);
        snprintf(name, sizeof name, "W PSRAM, -> int8 SRAM, %d core", cores); report(name, c, macs);
        if (memcmp(out8, ref8, T * N)) printf("  !! int8 output mismatch\n");
        job.mode = 3; memset(out8, 0, T * N); c = time_gemm(&job, cores);
        snprintf(name, sizeof name, "W %s, fused bias+int8 asm, %d core", "PSRAM", cores); report(name, c, macs);
        if (memcmp(out8, ref83, T * N)) {
            int i = 0; while (out8[i] == ref83[i]) i++;
            printf("  !! fused int8 mismatch at %d (t=%d n=%d): got %d want %d\n", i, i / N, i % N, out8[i], ref83[i]);
        }
    }
    heap_caps_free(out32);
    heap_caps_free(X); heap_caps_free(W); heap_caps_free(Wp_ps); heap_caps_free(ref); heap_caps_free(ref8);
    heap_caps_free(mult); heap_caps_free(out8); heap_caps_free(mhi); heap_caps_free(bias);
    heap_caps_free(bias_raw); heap_caps_free(ref83);
}

// ---------------------------------------------------------------- fp32 baseline
static void bench_gemm_f32(void) {
    const int T = 16, N = 128, K = 192;
    printf("\n== fp32 matmul, C, all in SRAM: T=%d N=%d K=%d ==\n", T, N, K);
    float *X = alloc(T * K * 4, SRAM), *W = alloc(N * K * 4, SRAM), *O = alloc(T * N * 4, SRAM);
    for (int i = 0; i < T * K; i++) X[i] = rand_s8(64) / 64.0f;
    for (int i = 0; i < N * K; i++) W[i] = rand_s8(64) / 64.0f;
    for (int rep = 0; rep < 2; rep++) {
        uint32_t t0 = now();
        for (int t = 0; t < T; t++)
            for (int n = 0; n < N; n++) {
                const float *x = X + t * K, *w = W + n * K;
                float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                for (int k = 0; k < K; k += 4) { s0 += x[k] * w[k]; s1 += x[k + 1] * w[k + 1]; s2 += x[k + 2] * w[k + 2]; s3 += x[k + 3] * w[k + 3]; }
                O[t * N + n] = s0 + s1 + s2 + s3;
            }
        if (rep) report("fp32 C dot, 4 accumulators", now() - t0, (double)T * N * K);
    }
    heap_caps_free(X); heap_caps_free(W); heap_caps_free(O);
}

// ---------------------------------------------------------------- memory
static void bench_memory(void) {
    printf("\n== memory read bandwidth ==\n");
    const int PS = 2 << 20, SR = 32 << 10;
    uint8_t *ps = alloc(PS, PSRAM), *sr = alloc(SR, SRAM);
    for (int i = 0; i < PS; i += 4) *(uint32_t *)(ps + i) = i;
    for (int rep = 0; rep < 2; rep++) {
        uint32_t t0 = now();
        for (int off = 0; off < PS; off += SR) memcpy(sr, ps + off, SR);
        uint32_t c1 = now() - t0;
        t0 = now();
        read_vld(ps, PS / 16);
        uint32_t c2 = now() - t0;
        t0 = now();
        volatile uint32_t sum = 0; uint32_t s = 0;
        for (int i = 0; i < PS; i += 4) s += *(volatile uint32_t *)(ps + i);
        sum = s; (void)sum;
        uint32_t c3 = now() - t0;
        t0 = now();
        for (int r = 0; r < 64; r++) read_vld(sr, SR / 16);
        uint32_t c4 = now() - t0;
        if (rep) {
            printf("  PSRAM->SRAM memcpy      %7.1f MB/s\n", (double)PS / c1 * MHZ);
            printf("  PSRAM 128-bit loads     %7.1f MB/s\n", (double)PS / c2 * MHZ);
            printf("  PSRAM 32-bit C loads    %7.1f MB/s\n", (double)PS / c3 * MHZ);
            printf("  SRAM 128-bit loads      %7.1f MB/s\n", (double)SR * 64 / c4 * MHZ);
        }
    }
    heap_caps_free(ps); heap_caps_free(sr);
}

// ---------------------------------------------------------------- scalar math
static void bench_math(void) {
    printf("\n== libm per call (cycles) ==\n");
    const int n = 4096;
    float *a = alloc(n * 4, SRAM);
    for (int i = 0; i < n; i++) a[i] = (i - n / 2) * 0.003f;
    volatile float sink = 0;
    struct { const char *name; float (*f)(float); } fns[] = {{"expf", expf}, {"sinf", sinf}, {"cosf", cosf}, {"logf", logf}, {"tanhf", tanhf}};
    for (int f = 0; f < 5; f++) {
        float s = 0;
        uint32_t t0 = now();
        for (int i = 0; i < n; i++) s += fns[f].f(f == 3 ? fabsf(a[i]) + 1e-3f : a[i]);
        uint32_t c = now() - t0;
        sink = s;
        printf("  %-6s %6.1f\n", fns[f].name, (double)c / n);
    }
    (void)sink;
    heap_caps_free(a);
}

void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(1500));   // let the USB console attach
    printf("\n#### tiny-voca bench start ####\n");
    printf("cpu %d MHz, core %d\n", MHZ, xPortGetCoreID());
    go_sem = xSemaphoreCreateBinary();
    done_sem = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCore(worker, "worker", 4096, NULL, configMAX_PRIORITIES - 1, NULL, 1);
#ifdef RUN_BASELINES
    bench_memory();
    bench_math();
    bench_gemm_f32();
#endif
    bench_probe();
    bench_gemm_s8();
    printf("\n#### tiny-voca bench done ####\n");
}
