// ESP32-S3 PIE kernels for the tiny-voca engine. Vector pointers must be 16-byte
// aligned; kernels read up to 48 bytes past their inputs (buffers carry slack).
    .text

// void dot_rows_s8_accx(const int8_t *x, const int8_t *W, int32_t *out, int N, int k16)
// out[n] = dot(x[0:K], W[n*K : n*K+K]), K = 16*k16, exact (40-bit ACCX, low 32 bits returned).
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

// void qacc_mm_s8(const int8_t *x, const int8_t *wblk, uint8_t *raw, int k32)
// 16 outputs: QACC is loaded from raw (64-byte layout: QACC_L low128, high32, QACC_H low128,
// high32, each in a 16-byte slot), QACC[j] += sum_k wblk[k*16+j] * x[k] for K = 32*k32,
// then stored back to raw. Weights loaded 3 instructions ahead, inputs one chunk ahead.
    .align 4
    .global qacc_mm_s8
    .type qacc_mm_s8, @function
qacc_mm_s8:
    entry a1, 32
    mov a8, a4
    ee.ld.qacc_l.l.128.ip a8, 16
    ee.ld.qacc_l.h.32.ip a8, 16
    ee.ld.qacc_h.l.128.ip a8, 16
    ee.ld.qacc_h.h.32.ip a8, 16
    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a3, 16
    ee.vld.128.ip q2, a3, 16
    ee.vld.128.ip q3, a3, 16
    loopnez a5, .Lmm_end
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
.Lmm_end:
    ee.st.qacc_l.l.128.ip a4, 16
    ee.st.qacc_l.h.32.ip a4, 16
    ee.st.qacc_h.l.128.ip a4, 16
    ee.st.qacc_h.h.32.ip a4, 16
    retw
