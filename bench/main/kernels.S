// int8 matmul kernels for the ESP32-S3 PIE (128-bit SIMD).
// Every vector pointer must be 16-byte aligned. Kernels read 16 bytes past
// the end of their inputs (software pipelining), so buffers need 16 bytes of slack.

    .text

// int32 dot_rows_s8_accx(const int8_t *x, const int8_t *W, int32_t *out, int N, int k16)
// out[n] = dot(x[0:K], W[n*K : n*K+K]) for n < N, K = 16*k16. Uses the 40-bit ACCX.
    .align 4
    .global dot_rows_s8_accx
    .type dot_rows_s8_accx, @function
dot_rows_s8_accx:
    entry a1, 32
    mov a8, a2
.Lrows_n:
    mov a2, a8
    ee.zero.accx
    ee.vld.128.ip q0, a2, 16
    loopnez a6, .Lrows_k
    ee.vld.128.ip q1, a3, 16
    ee.vmulas.s8.accx.ld.ip q0, a2, 16, q0, q1
.Lrows_k:
    rur.accx_0 a9
    s32i a9, a4, 0
    addi a4, a4, 4
    addi a5, a5, -1
    bnez a5, .Lrows_n
    retw

// void dot_cols_s8_accx(const int8_t *X, const int8_t *w, int32_t *out, int T, int k16, int out_stride_bytes)
// One weight row against T activation rows (X rows contiguous, K bytes each):
// out[t * stride] = dot(X[t], w). The weight row stays in cache while t runs.
    .align 4
    .global dot_cols_s8_accx
    .type dot_cols_s8_accx, @function
dot_cols_s8_accx:
    entry a1, 32
    mov a8, a3
.Lcols_t:
    mov a3, a8
    ee.zero.accx
    ee.vld.128.ip q0, a2, 16
    loopnez a6, .Lcols_k
    ee.vld.128.ip q1, a3, 16
    ee.vmulas.s8.accx.ld.ip q0, a2, 16, q0, q1
.Lcols_k:
    addi a2, a2, -16
    rur.accx_0 a9
    s32i a9, a4, 0
    add a4, a4, a7
    addi a5, a5, -1
    bnez a5, .Lcols_t
    retw

// void qacc_block_s8(const int8_t *x, const int8_t *wblk, uint8_t *qacc_out, int k16)
// 16 outputs at once: QACC[j] += sum_k wblk[k*16 + j] * x[k]. wblk is a [K][16]
// tile (16 output channels interleaved). QACC lanes are 20-bit; the raw
// accumulator is stored as 4 x 16 bytes: QACC_L low128, QACC_L high32, QACC_H low128, QACC_H high32.
    .align 4
    .global qacc_block_s8
    .type qacc_block_s8, @function
qacc_block_s8:
    entry a1, 32
    ee.zero.qacc
    ee.vld.128.ip q1, a3, 16
    loopnez a5, .Lqacc_k
    ee.vld.128.ip q0, a2, 16
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q1, q0, 0
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 1
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q1, q0, 2
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 3
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q1, q0, 4
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 5
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q1, q0, 6
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 7
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q1, q0, 8
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 9
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q1, q0, 10
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 11
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q1, q0, 12
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 13
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q1, q0, 14
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 15
.Lqacc_k:
    ee.st.qacc_l.l.128.ip a4, 16
    ee.st.qacc_l.h.32.ip a4, 16
    ee.st.qacc_h.l.128.ip a4, 16
    ee.st.qacc_h.h.32.ip a4, 16
    retw

// void read_vld(const void *p, int n16): stream n16 x 16 bytes through q0 (bandwidth test).
    .align 4
    .global read_vld
    .type read_vld, @function
read_vld:
    entry a1, 32
    loopnez a3, .Lread_end
    ee.vld.128.ip q0, a2, 16
.Lread_end:
    retw

// void qacc_block_s8_v2(const int8_t *x, const int8_t *wblk, uint8_t *qacc_out, int k32)
// Same result as qacc_block_s8, K = 32*k32. Weights are loaded 3 instructions
// before use (q1..q4 rotate) and the next 16 inputs are loaded one chunk ahead
// (q0/q5 alternate), so no instruction waits on a load. Over-reads 48 bytes of
// weights and 16 bytes of inputs.
    .align 4
    .global qacc_block_s8_v2
    .type qacc_block_s8_v2, @function
qacc_block_s8_v2:
    entry a1, 32
    ee.zero.qacc
    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a3, 16
    ee.vld.128.ip q2, a3, 16
    ee.vld.128.ip q3, a3, 16
    loopnez a5, .Lv2_end
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q0, 0
    ee.vld.128.ip q5, a2, 16
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 1
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q0, 2
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q0, 3
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q0, 4
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 5
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q0, 6
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q0, 7
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q0, 8
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 9
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q0, 10
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q0, 11
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q0, 12
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 13
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q0, 14
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q0, 15
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q5, 0
    ee.vld.128.ip q0, a2, 16
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q5, 1
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q5, 2
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q5, 3
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q5, 4
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q5, 5
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q5, 6
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q5, 7
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q5, 8
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q5, 9
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q5, 10
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q5, 11
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q5, 12
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q5, 13
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q5, 14
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q5, 15
.Lv2_end:
    ee.st.qacc_l.l.128.ip a4, 16
    ee.st.qacc_l.h.32.ip a4, 16
    ee.st.qacc_h.l.128.ip a4, 16
    ee.st.qacc_h.h.32.ip a4, 16
    retw

// void qacc_probe(const int8_t *w, const int8_t *x, int n, uint8_t *out)
// QACC += w*x (lane-wise) n times, then dumps: raw @0, srcmb.s16 (shift 0) @64,
// raw again @80, srcmb.s8 (shift 8) @144, raw again @160. Shows overflow and
// whether srcmb changes QACC.
    .align 4
    .global qacc_probe
    .type qacc_probe, @function
qacc_probe:
    entry a1, 32
    ee.zero.qacc
    ee.vld.128.ip q0, a2, 0
    ee.vld.128.ip q1, a3, 0
    loopnez a4, .Lprobe_end
    ee.vmulas.s8.qacc q0, q1
.Lprobe_end:
    movi a8, 0
    movi a9, 8
    ee.st.qacc_l.l.128.ip a5, 16
    ee.st.qacc_l.h.32.ip a5, 16
    ee.st.qacc_h.l.128.ip a5, 16
    ee.st.qacc_h.h.32.ip a5, 16
    ee.srcmb.s16.qacc q2, a8, 0
    ee.vst.128.ip q2, a5, 16
    ee.st.qacc_l.l.128.ip a5, 16
    ee.st.qacc_l.h.32.ip a5, 16
    ee.st.qacc_h.l.128.ip a5, 16
    ee.st.qacc_h.h.32.ip a5, 16
    ee.srcmb.s8.qacc q3, a9, 0
    ee.vst.128.ip q3, a5, 16
    ee.st.qacc_l.l.128.ip a5, 16
    ee.st.qacc_l.h.32.ip a5, 16
    ee.st.qacc_h.l.128.ip a5, 16
    ee.st.qacc_h.h.32.ip a5, 16
    retw

// void qacc_block_s8_q8(const int8_t *x, const int8_t *wblk, int8_t *out, int k32,
//                       const int32_t *mult, const uint8_t *bias_raw)
// 16 outputs, fully fused: QACC starts from the bias (64-byte raw layout, see
// pack_qacc), runs the v2 pipelined loop, then each 20-bit lane j becomes
// out[j] = clamp_int8((acc_j * mult[j]) >> 32) without leaving registers.
    .align 4
    .global qacc_block_s8_q8
    .type qacc_block_s8_q8, @function
qacc_block_s8_q8:
    entry a1, 128   // scratch at a1+32..a1+95 stays below the caller's spill area (top 16 bytes)
    ee.ld.qacc_l.l.128.ip a7, 16
    ee.ld.qacc_l.h.32.ip a7, 16
    ee.ld.qacc_h.l.128.ip a7, 16
    ee.ld.qacc_h.h.32.ip a7, 16
    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a3, 16
    ee.vld.128.ip q2, a3, 16
    ee.vld.128.ip q3, a3, 16
    loopnez a5, .Lq8_end
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q0, 0
    ee.vld.128.ip q5, a2, 16
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 1
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q0, 2
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q0, 3
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q0, 4
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 5
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q0, 6
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q0, 7
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q0, 8
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 9
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q0, 10
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q0, 11
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q0, 12
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q0, 13
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q0, 14
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q0, 15
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q5, 0
    ee.vld.128.ip q0, a2, 16
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q5, 1
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q5, 2
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q5, 3
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q5, 4
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q5, 5
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q5, 6
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q5, 7
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q5, 8
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q5, 9
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q5, 10
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q5, 11
    ee.vsmulas.s8.qacc.ld.incp q4, a3, q1, q5, 12
    ee.vsmulas.s8.qacc.ld.incp q1, a3, q2, q5, 13
    ee.vsmulas.s8.qacc.ld.incp q2, a3, q3, q5, 14
    ee.vsmulas.s8.qacc.ld.incp q3, a3, q4, q5, 15
.Lq8_end:
    addi a8, a1, 32
    ee.st.qacc_l.l.128.ip a8, 16
    ee.st.qacc_l.h.32.ip a8, 16
    ee.st.qacc_h.l.128.ip a8, 16
    ee.st.qacc_h.h.32.ip a8, 16
    addi a8, a1, 32
    l32i a9, a8, 0
    l32i a10, a8, 4
    l32i a11, a8, 8
    l32i a12, a8, 12
    l32i a13, a8, 16
    l32i a2, a6, 0
    sext a14, a9, 19
    mulsh a14, a14, a2
    clamps a14, a14, 7
    s8i a14, a4, 0
    l32i a3, a6, 4
    ssai 20
    src a15, a10, a9
    sext a15, a15, 19
    mulsh a15, a15, a3
    clamps a15, a15, 7
    s8i a15, a4, 1
    l32i a2, a6, 8
    srli a14, a10, 8
    sext a14, a14, 19
    mulsh a14, a14, a2
    clamps a14, a14, 7
    s8i a14, a4, 2
    l32i a3, a6, 12
    ssai 28
    src a15, a11, a10
    sext a15, a15, 19
    mulsh a15, a15, a3
    clamps a15, a15, 7
    s8i a15, a4, 3
    l32i a2, a6, 16
    ssai 16
    src a14, a12, a11
    sext a14, a14, 19
    mulsh a14, a14, a2
    clamps a14, a14, 7
    s8i a14, a4, 4
    l32i a3, a6, 20
    srli a15, a12, 4
    sext a15, a15, 19
    mulsh a15, a15, a3
    clamps a15, a15, 7
    s8i a15, a4, 5
    l32i a2, a6, 24
    ssai 24
    src a14, a13, a12
    sext a14, a14, 19
    mulsh a14, a14, a2
    clamps a14, a14, 7
    s8i a14, a4, 6
    l32i a3, a6, 28
    srai a15, a13, 12
    mulsh a15, a15, a3
    clamps a15, a15, 7
    s8i a15, a4, 7
    l32i a9, a8, 32
    l32i a10, a8, 36
    l32i a11, a8, 40
    l32i a12, a8, 44
    l32i a13, a8, 48
    l32i a2, a6, 32
    sext a14, a9, 19
    mulsh a14, a14, a2
    clamps a14, a14, 7
    s8i a14, a4, 8
    l32i a3, a6, 36
    ssai 20
    src a15, a10, a9
    sext a15, a15, 19
    mulsh a15, a15, a3
    clamps a15, a15, 7
    s8i a15, a4, 9
    l32i a2, a6, 40
    srli a14, a10, 8
    sext a14, a14, 19
    mulsh a14, a14, a2
    clamps a14, a14, 7
    s8i a14, a4, 10
    l32i a3, a6, 44
    ssai 28
    src a15, a11, a10
    sext a15, a15, 19
    mulsh a15, a15, a3
    clamps a15, a15, 7
    s8i a15, a4, 11
    l32i a2, a6, 48
    ssai 16
    src a14, a12, a11
    sext a14, a14, 19
    mulsh a14, a14, a2
    clamps a14, a14, 7
    s8i a14, a4, 12
    l32i a3, a6, 52
    srli a15, a12, 4
    sext a15, a15, 19
    mulsh a15, a15, a3
    clamps a15, a15, 7
    s8i a15, a4, 13
    l32i a2, a6, 56
    ssai 24
    src a14, a13, a12
    sext a14, a14, 19
    mulsh a14, a14, a2
    clamps a14, a14, 7
    s8i a14, a4, 14
    l32i a3, a6, 60
    srai a15, a13, 12
    mulsh a15, a15, a3
    clamps a15, a15, 7
    s8i a15, a4, 15
    retw
