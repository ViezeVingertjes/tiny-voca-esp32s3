// Compute kernels behind the engine, and what the engine's files share among themselves.
// Two backends with identical integer results:
//   tv_kernels_ref.c  portable C; emulates the S3 QACC (20-bit saturating lanes)
//   tv_kernels_s3.c   ESP32-S3: PIE assembly (kernels_s3.S), both cores; built by firmware/
#pragma once
#include "tv.h"

#ifdef __cplusplus
extern "C" {
#endif

// 16-byte aligned, with at least 64 bytes of readable slack after n (kernels over-read).
void *tv_alloc(size_t n);
// The same, in faster memory when there is room (small buffers the kernels read a lot).
void *tv_alloc_fast(size_t n);
void tv_free(void *p);

void tv_kernels_init(void);
// Per-linear preparation after loading (e.g. packing QACC preloads). Returns 0 on success.
int tv_linear_prepare(tv_linear *L);
void tv_linear_release(tv_linear *L);

// What happens to each output y = acc * (L->in_scale * L->ws[n]) + L->bias[n]:
enum { TV_EPI_STORE,        // out (float*)[t][n] = y
       TV_EPI_ADD,          // out (float*)[t][n] += y            (residual stream)
       TV_EPI_Q8 };         // out (int8_t*)[t][n] = lut[q + 128] or q, q = round(y * inv) clamped to +-127
typedef struct {
    int mode;
    float inv;
    const int8_t *lut;
} tv_epilogue;

// acc = sum_k x[t][k] * w[n][k] with QACC semantics (per-chunk preload, 20-bit lanes), then the
// epilogue. x: [T][K] rows, K % 32 == 0. Returns the number of saturated lanes seen (must be 0).
long tv_linear_run(const tv_linear *L, const int8_t *x, int T, void *out, const tv_epilogue *e);

// The engine's float -> int8 rounding (half to even), shared by both backends.
static inline int8_t tv_quant(float x, float inv) {
    float q = x * inv;
    q = q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q;
    return (int8_t)((int32_t)(q + 12582912.0f) - 12582912);
}

// Runs fn(a) and fn(b), on the two cores where there are two. Not re-entrant: fn must not call
// tv_parallel2 (tv_linear_run does).
void tv_parallel2(void (*fn)(void *), void *a, void *b);

// Attention over one head (dimension 48), dots d[t] = q . K[t*48 : t*48+48] for t < n:
// max(m, d[0..n)).
int32_t tv_attn_max48(const int8_t *q, const int8_t *K, int n, int32_t m);
// w[t] = lut[min(m - d[t], lim)] for t < n (lut[lim] must be 0); returns the sum of the w[t].
int32_t tv_attn_weights48(const int8_t *q, const int8_t *K, int n, int32_t m, const uint8_t *lut, uint32_t lim, int8_t *w);
// acc[d] += sum_t w[t] * v[d*128 + t] for d < 48, t < 128 (one key chunk of transposed values).
void tv_attn_pv48(const int8_t *w, const int8_t *v, int32_t *acc);

uint64_t tv_time_us(void);

// ---------------------------------------------------------------- engine internals
// Profile: microseconds per op category, reported by tv_profile_report.
enum { TV_P_MATMUL, TV_P_ATTN_DOT, TV_P_ATTN_PV, TV_P_NORM, TV_P_DW, TV_P_QUANT, TV_P_GELU, TV_P_ROPE,
       TV_P_VOC_CEP, TV_P_VOC_SHAPE, TV_P_COUNT };
extern uint64_t tv_prof[TV_P_COUNT];
#define TV_PROF(id, stmt) do { uint64_t tv_t0_ = tv_time_us(); stmt; tv_prof[id] += tv_time_us() - tv_t0_; } while (0)

// The vocoder (tv_dsp.c). Frame f of harmonic [T][TV_HARM_PARAMS], noise_params [T][TV_NOISE_PARAMS]
// and f0 [T] is at f % ring (ring 0: T, whole arrays). need (may be NULL) is called before each
// block with the number of leading frames that must be final by then, so the caller can compute
// them as they are asked for.
typedef void (*tv_need_fn)(void *user, int frames);
void tv_vocoder_stream(const float *harmonic, const float *noise_params, const float *f0, const float *noise,
                       uint64_t seed, int T, int ring, tv_audio_fn sink, void *user, tv_need_fn need, void *need_user);

#ifdef __cplusplus
}
#endif
