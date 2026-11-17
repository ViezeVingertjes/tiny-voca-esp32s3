// Compute kernels behind the engine. Two backends with identical integer results:
//   tv_kernels_ref.c  portable C; emulates the S3 QACC (20-bit saturating lanes)
//   firmware/main/tv_kernels_s3.c  ESP32-S3 PIE assembly, both cores
#pragma once
#include "tv.h"

// 16-byte aligned, with at least 64 bytes of readable slack after n (kernels over-read).
void *tv_alloc(size_t n);
void tv_free(void *p);

void tv_kernels_init(void);
// Per-linear preparation after loading (e.g. packing QACC preloads). Returns 0 on success.
int tv_linear_prepare(tv_linear *L);
void tv_linear_release(tv_linear *L);

// y[t][n] = acc[t][n] * (L->in_scale * L->ws[n]) + L->bias[n], acc = sum_k x[t][k] * w[n][k] with
// QACC semantics (per-chunk preload, 20-bit lanes). x: [T][K] rows, K % 32 == 0.
// Returns the number of saturated lanes seen (must be 0).
long tv_linear_f32(const tv_linear *L, const int8_t *x, int T, float *y);

// Runs fn(a) and fn(b), on the two cores where there are two. fn must not call tv_linear_f32.
void tv_parallel2(void (*fn)(void *), void *a, void *b);

// Exact int32 dot products: out[n] = sum_k x[k] * W[n*K + k]. K % 16 == 0.
void tv_dot_rows_i8(const int8_t *x, const int8_t *W, int N, int K, int32_t *out);
