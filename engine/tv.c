// tiny-voca engine: model loading and the forward pass. Numerics follow
// tools/tvq.py (the golden model); see that file for the specification.
#include "tv.h"
#include "tv_kernels.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

#define GELU_MIN (-0.16997124f)
#define VOCAB_UNK 4
#define FIRST_PHONE 17
#define FIRST_VOWEL 43

// ---------------------------------------------------------------- container
_Static_assert(sizeof(tv_entry) == 96, "TVQ1 entries are 96 bytes");

int tv_file_open(tv_file *f, const void *data, size_t size) {
    const uint8_t *p = data;
    if (size < 16 || memcmp(p, "TVQ1", 4)) return -1;
    uint32_t version, count;
    memcpy(&version, p + 4, 4);
    memcpy(&count, p + 8, 4);
    if (version != 1 || count > (size - 16) / sizeof(tv_entry)) return -1;
    f->base = p;
    f->entries = (const tv_entry *)(p + 16);
    f->count = count;
    for (uint32_t i = 0; i < count; i++) {     // the kernels need 16-byte aligned tensors
        const tv_entry *e = &f->entries[i];
        if (e->offset % 16 || e->offset > size || e->nbytes > size - e->offset) return -1;
    }
    return 0;
}

const tv_entry *tv_file_find(const tv_file *f, const char *name) {
    for (uint32_t i = 0; i < f->count; i++)
        if (!strncmp(f->entries[i].name, name, 64)) return &f->entries[i];
    return NULL;
}

const void *tv_file_get(const tv_file *f, const char *name, uint32_t dtype, uint32_t nelem) {
    const tv_entry *e = tv_file_find(f, name);
    static const uint32_t size[] = {4, 1, 4};
    if (!e || e->dtype != dtype || e->nbytes != nelem * size[dtype]) return NULL;
    return f->base + e->offset;
}

// ---------------------------------------------------------------- model loading
typedef struct {
    tv_model *m;
    int failed;
} loader;

static void fail(loader *L, const char *fmt, const char *what) {
    if (L->failed) return;
    snprintf(L->m->error, sizeof L->m->error, fmt, what);
    L->failed = 1;
}

// The tensor named by fmt, ...; a missing or malformed one fails the load.
static const void *get(loader *L, uint32_t dtype, uint32_t nelem, const char *fmt, ...) {
    char name[96];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(name, sizeof name, fmt, ap);
    va_end(ap);
    if (n < 0 || n >= (int)sizeof name) { fail(L, "tensor name too long (%s)", fmt); return NULL; }
    const void *p = tv_file_get(&L->m->file, name, dtype, nelem);
    if (!p) fail(L, "missing or malformed tensor %s", name);
    return p;
}

static float scale_of(loader *L, const char *prefix, const char *point) {
    const float *v = get(L, TV_F32, 1, "%s.%s.max", prefix, point);
    return v ? (float)((double)*v / 127.0) : 1.0f;
}

// The int8 linear `<prefix><part>` (part may be "").
static void load_linear(loader *L, tv_linear *lin, const char *prefix, const char *part, int N, int K, float in_scale,
                        int chunks) {
    int npad = (N + 15) / 16 * 16;
    lin->N = N, lin->K = K, lin->in_scale = in_scale, lin->chunks = 1, lin->pre = NULL;
    lin->wq = get(L, TV_I8, npad * K, "%s%s.wq", prefix, part);
    lin->ws = get(L, TV_F32, N, "%s%s.ws", prefix, part);
    lin->b = get(L, TV_F32, N, "%s%s.b", prefix, part);
    if (chunks > 1) {
        lin->pre = get(L, TV_I32, chunks * N, "%s%s.pre", prefix, part);
        lin->chunks = chunks;
    }
    lin->bias = calloc(N, sizeof(float));
    if (!lin->bias) fail(L, "out of memory loading %s", prefix);
    if (L->failed) return;
    memcpy(lin->bias, lin->b, N * sizeof(float));
    if (tv_linear_prepare(lin)) fail(L, "kernel preparation failed for %s", prefix);
}

static float gelu(float x) { return 0.5f * x * (1.0f + erff(x * (float)M_SQRT1_2)); }

static void load_block(loader *L, tv_block *b, const char *prefix) {
    b->norm1_w = get(L, TV_F32, TV_C, "%s.norm1.weight", prefix);
    b->norm1_b = get(L, TV_F32, TV_C, "%s.norm1.bias", prefix);
    b->norm2_w = get(L, TV_F32, TV_C, "%s.norm2.weight", prefix);
    b->norm2_b = get(L, TV_F32, TV_C, "%s.norm2.bias", prefix);
    b->dw_w = get(L, TV_F32, TV_C * TV_KERNEL, "%s.depthwise.weight", prefix);
    b->dw_b = get(L, TV_F32, TV_C, "%s.depthwise.bias", prefix);
    b->s_ln1 = scale_of(L, prefix, "ln1");
    b->s_q = scale_of(L, prefix, "q");
    b->s_k = scale_of(L, prefix, "k");
    b->s_v = scale_of(L, prefix, "v");
    b->s_attn = scale_of(L, prefix, "attn");
    b->s_dw = scale_of(L, prefix, "dw");
    b->s_pre = scale_of(L, prefix, "pre");
    const float *gmax = get(L, TV_F32, 1, "%s.gelu.max", prefix);   // the asymmetric GELU codes span [GELU_MIN, max]
    b->s_gelu = gmax ? (float)(((double)*gmax - (double)GELU_MIN) / 255.0) : 1.0f;
    load_linear(L, &b->qkv, prefix, ".attention.qkv", 3 * TV_C, TV_C, b->s_ln1, 1);
    load_linear(L, &b->out, prefix, ".attention.out", TV_C, TV_C, b->s_attn, 1);
    load_linear(L, &b->expand, prefix, ".expand", TV_FF, TV_C, b->s_dw, 1);
    load_linear(L, &b->project, prefix, ".project", TV_C, TV_FF, b->s_gelu, 2);
    if (L->failed) return;
    // GELU table: pre code c -> asymmetric gelu code, exactly as the golden model rounds.
    for (int c = -128; c <= 127; c++) {
        float g = gelu((float)(c < -127 ? -127 : c) * b->s_pre);
        float code = rintf((g - GELU_MIN) / b->s_gelu);
        code = code < 0 ? 0 : code > 255 ? 255 : code;
        b->gelu_lut[c + 128] = (int8_t)(code - 128);
    }
    // Attention weights depend only on the integer gap to the row's max dot product.
    double score_scale = (double)b->s_q * b->s_k / sqrt((double)TV_HEAD_DIM);
    b->softmax_n = (int)ceil(log(254.0) / score_scale) + 1;      // 127 * exp(-gap * s) < 0.5 beyond
    b->softmax_lut = malloc(b->softmax_n + 1);         // small and read in the attention loops: internal RAM
    if (!b->softmax_lut) { fail(L, "out of memory loading %s", prefix); return; }
    double v = 127.0, step = exp(-score_scale);             // 127 * exp(-g * s) by repeated multiplication
    for (int g = 0; g < b->softmax_n; g++, v *= step) b->softmax_lut[g] = (uint8_t)lrint(v);
    b->softmax_lut[b->softmax_n] = 0;                       // gaps are clamped to softmax_n
    // project reads codes q standing for (q + 128) * s_gelu + GELU_MIN: fold the constant part.
    tv_linear *p = &b->project;
    for (int n = 0; n < p->N; n++) {
        long rowsum = 0;
        const int8_t *w = p->wq + (n / 16) * p->K * 16 + n % 16;
        for (int k = 0; k < p->K; k++) rowsum += w[k * 16];
        p->bias[n] = p->b[n] + p->ws[n] * (128.0f * b->s_gelu + GELU_MIN) * (float)rowsum;
    }
}

static void load_stack(loader *L, tv_stack *s, tv_block *storage, const char *name, int count) {
    char prefix[64];
    s->blocks = storage;
    s->count = count;
    for (int i = 0; i < count; i++) {
        snprintf(prefix, sizeof prefix, "%s.blocks.%d", name, i);
        load_block(L, &storage[i], prefix);
    }
    s->norm_w = get(L, TV_F32, TV_C, "%s.norm.weight", name);
    s->norm_b = get(L, TV_F32, TV_C, "%s.norm.bias", name);
}

int tv_model_init(tv_model *m, const void *data, size_t size) {
    memset(m, 0, sizeof *m);
    tv_kernels_init();
    if (tv_file_open(&m->file, data, size)) {
        snprintf(m->error, sizeof m->error, "not a TVQ1 file");
        return -1;
    }
    loader L = {m, 0};
    m->tokens = get(&L, TV_F32, TV_VOCAB * TV_C, "tokens.weight");
    m->stress = get(&L, TV_F32, TV_STRESS * TV_C, "stress.weight");
    m->voices = get(&L, TV_F32, TV_VOICES * TV_C, "voices.weight");
    m->frame_voices = get(&L, TV_F32, TV_VOICES * TV_C, "frame_voices.weight");
    tv_block *blocks = m->block_storage;
    load_stack(&L, &m->encoder, blocks, "encoder", TV_ENCODER_BLOCKS);
    load_stack(&L, &m->prosody, blocks + TV_ENCODER_BLOCKS, "prosody_stack", TV_PROSODY_BLOCKS);
    load_stack(&L, &m->acoustic, blocks + TV_ENCODER_BLOCKS + TV_PROSODY_BLOCKS, "acoustic_stack", TV_ACOUSTIC_BLOCKS);
    load_linear(&L, &m->dur1, "duration.conv1", "", TV_PRED, 3 * TV_C, scale_of(&L, "dur", "in"), 1);
    load_linear(&L, &m->dur2, "duration.conv2", "", TV_PRED, 3 * TV_PRED, scale_of(&L, "dur", "h1"), 1);
    m->dnorm1_w = get(&L, TV_F32, TV_PRED, "duration.norm1.weight");
    m->dnorm1_b = get(&L, TV_F32, TV_PRED, "duration.norm1.bias");
    m->dnorm2_w = get(&L, TV_F32, TV_PRED, "duration.norm2.weight");
    m->dnorm2_b = get(&L, TV_F32, TV_PRED, "duration.norm2.bias");
    m->dur_out_w = get(&L, TV_F32, 2 * TV_PRED, "duration.out.weight");
    m->dur_out_b = get(&L, TV_F32, 2, "duration.out.bias");
    m->frame_pos_w = get(&L, TV_F32, TV_C * 3, "frame_position.weight");
    m->frame_pos_b = get(&L, TV_F32, TV_C, "frame_position.bias");
    m->prosody_head_w = get(&L, TV_F32, 3 * TV_C, "prosody_head.weight");
    m->prosody_head_b = get(&L, TV_F32, 3, "prosody_head.bias");
    m->prosody_proj_w = get(&L, TV_F32, TV_C * 3, "prosody_projection.weight");
    m->prosody_proj_b = get(&L, TV_F32, TV_C, "prosody_projection.bias");
    m->s_voc = scale_of(&L, "voc", "in");
    load_linear(&L, &m->harmonic, "vocoder.harmonic", "", TV_HARM_PARAMS, TV_C, m->s_voc, 1);
    load_linear(&L, &m->noise, "vocoder.noise", "", TV_NOISE_PARAMS, TV_C, m->s_voc, 1);
    return L.failed ? -1 : 0;
}

static void free_linear(tv_linear *l) { tv_linear_release(l); free(l->bias); l->bias = NULL; }

void tv_model_free(tv_model *m) {
    for (int i = 0; i < TV_BLOCKS; i++) {
        tv_block *b = &m->block_storage[i];
        free_linear(&b->qkv); free_linear(&b->out); free_linear(&b->expand); free_linear(&b->project);
        free(b->softmax_lut); b->softmax_lut = NULL;
    }
    free_linear(&m->dur1); free_linear(&m->dur2); free_linear(&m->harmonic); free_linear(&m->noise);
}

// ---------------------------------------------------------------- profiling
uint64_t tv_prof[TV_P_COUNT];
static const char *const prof_names[TV_P_COUNT] = {"matmul (int8)", "attention pass 1", "attention pass 2",
    "layernorm", "depthwise conv", "quantize", "gelu table", "rotary + q/k/v codes", "vocoder cepstrum", "vocoder stft filter"};
void tv_profile_reset(void) { memset(tv_prof, 0, sizeof tv_prof); }
void tv_profile_report(void) {
    uint64_t total = 0;
    for (int i = 0; i < TV_P_COUNT; i++) total += tv_prof[i];
    for (int i = 0; i < TV_P_COUNT; i++)
        printf("  %-22s %9.1f ms  %5.1f%%\n", prof_names[i], tv_prof[i] / 1000.0, total ? 100.0 * tv_prof[i] / total : 0.0);
    printf("  %-22s %9.1f ms\n", "(sum of the above)", total / 1000.0);
}

// ---------------------------------------------------------------- ops
typedef struct {
    const tv_options *opt;
    long saturations;
} ctx;

static void dbg(ctx *c, const char *name, int dtype, const void *data, int rows, int cols) {
    if (c->opt->debug) c->opt->debug(c->opt->debug_user, name, dtype, data, rows, cols);
}

static void quantize(const float *x, int n, float scale, int8_t *q) {
    uint64_t t0 = tv_time_us();
    const float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) q[i] = tv_quant(x[i], inv);
    tv_prof[TV_P_QUANT] += tv_time_us() - t0;
}

// y = dequantized matmul + bias (float output).
static void linear_f(ctx *c, const tv_linear *L, const int8_t *x, int T, float *y) {
    tv_epilogue e = {TV_EPI_STORE, 0, NULL};
    TV_PROF(TV_P_MATMUL, c->saturations += tv_linear_run(L, x, T, y, &e));
}

// residual[t][n] += linear output
static void linear_add(ctx *c, const tv_linear *L, const int8_t *x, int T, float *residual) {
    tv_epilogue e = {TV_EPI_ADD, 0, NULL};
    TV_PROF(TV_P_MATMUL, c->saturations += tv_linear_run(L, x, T, residual, &e));
}

// out = lut[quantized linear output] (int8)
static void linear_q8(ctx *c, const tv_linear *L, const int8_t *x, int T, int8_t *out, float scale, const int8_t *lut) {
    tv_epilogue e = {TV_EPI_Q8, 1.0f / scale, lut};
    TV_PROF(TV_P_MATMUL, c->saturations += tv_linear_run(L, x, T, out, &e));
}

// Mean and 1 / standard deviation of one LayerNorm row.
static inline float norm_stats(const float *x, int C, float *mean_out) {
    float mean = 0, var = 0;
    for (int i = 0; i < C; i++) mean += x[i];
    mean /= C;
    for (int i = 0; i < C; i++) var += (x[i] - mean) * (x[i] - mean);
    *mean_out = mean;
    return 1.0f / sqrtf(var / C + 1e-5f);
}

// LayerNorm straight to int8 codes (same arithmetic as layer_norm followed by quantize).
static void layer_norm_q8_rows(const float *x, int t_begin, int T, int C, const float *w, const float *b, float scale, int8_t *q) {
    const float qinv = 1.0f / scale;
    for (int t = t_begin; t < T; t++) {
        const float *xt = x + (size_t)t * C;
        int8_t *qt = q + (size_t)t * C;
        float mean, inv = norm_stats(xt, C, &mean);
        for (int i = 0; i < C; i++) {
            float y = (xt[i] - mean) * inv * w[i] + b[i];
            qt[i] = tv_quant(y, qinv);
        }
    }
}

// y may alias x: each value is read before it is written.
static void layer_norm_rows(const float *x, int t_begin, int T, int C, const float *w, const float *b, float *y) {
    for (int t = t_begin; t < T; t++) {
        const float *xt = x + (size_t)t * C;
        float *yt = y + (size_t)t * C;
        float mean, inv = norm_stats(xt, C, &mean);
        for (int i = 0; i < C; i++) yt[i] = (xt[i] - mean) * inv * w[i] + b[i];
    }
}

static void layer_norm(const float *x, int T, int C, const float *w, const float *b, float *y) {
    TV_PROF(TV_P_NORM, layer_norm_rows(x, 0, T, C, w, b, y));
}

static void windows3(const float *x, int T, int C, float *y) {
    for (int t = 0; t < T; t++)
        for (int j = 0; j < 3; j++) {
            int s = t + j - 1;
            float *dst = y + (size_t)t * 3 * C + j * C;
            if (s < 0 || s >= T) memset(dst, 0, C * sizeof(float));
            else memcpy(dst, x + (size_t)s * C, C * sizeof(float));
        }
}

// Rotary tables for positions 0 .. T-1: cos [T][24], then sin [T][24].
static float *rotary_table(int T) {
    const int half = TV_HEAD_DIM / 2;
    float *cos_t = tv_alloc((size_t)T * TV_HEAD_DIM * sizeof(float)), *sin_t = cos_t + (size_t)T * half;
    uint64_t t0 = tv_time_us();
    for (int i = 0; i < half; i++) {
        float inv = 1.0f / powf(10000.0f, (float)(2 * i) / (float)TV_HEAD_DIM);
        for (int t = 0; t < T; t++) {
            float a = (float)t * inv;
            cos_t[t * half + i] = cosf(a);
            sin_t[t * half + i] = sinf(a);
        }
    }
    tv_prof[TV_P_ROPE] += tv_time_us() - t0;
    return cos_t;
}

// ---------------------------------------------------------------- tiled transformer stack
// Each block runs over row tiles of TV_TILE positions. Only the residual stream (the caller's
// buffer) and the int8 queries, keys (per head) and values (per head, transposed) span the whole
// utterance: about 0.6 KB per position here plus the residual, instead of ~10 KB untiled.
#define TV_TILE 64
#define TV_KCH 128                  // keys per chunk: 6 KB of keys and 6 KB of values stay in cache

// An int8 debug tensor `<prefix>.<point>` [T][cols] assembled from tiles in *buf (debug runs only).
static void dbg_tile(ctx *c, uint8_t **buf, const char *prefix, const char *point, const int8_t *tile, int row0, int rows,
                     int cols, int T) {
    if (!c->opt->debug) return;
    if (!*buf) *buf = tv_alloc((size_t)T * cols);
    memcpy(*buf + (size_t)row0 * cols, tile, (size_t)rows * cols);
    if (row0 + rows == T) {
        char name[96];
        snprintf(name, sizeof name, "%s.%s", prefix, point);
        dbg(c, name, TV_I8, *buf, T, cols);
        tv_free(*buf);
        *buf = NULL;
    }
}

typedef struct {
    int T, nchunks;                 // nchunks = ceil(T / TV_KCH)
    int lookahead;                  // > 0: queries see keys up to their own position + lookahead
    int past;                       // > 0: and from their own position - past
    // Queries, keys and values are kept for the last `ring` key chunks (nchunks: the whole
    // utterance); position t is at t mod (ring * TV_KCH), so a chunk stays contiguous.
    int ring;
    int8_t *q8;                     // [ring * TV_KCH][192]
    int8_t *kh;                     // [heads][ring * TV_KCH][48]
    int8_t *vt;                     // [heads][ring][48][TV_KCH] values, transposed per key chunk
    int32_t *vacc[2];               // [TV_TILE][48] per core: weighted value sums of one head
    int32_t *cmax[2];               // [TV_TILE][nchunks] per core: max dot product per key chunk
    int8_t *w8[2];                  // [TV_TILE][TV_KCH] per core: attention weights of one key chunk
    const float *cos_t, *sin_t;     // [T][24] rotary tables (whole utterance)
    float *own_rotary;              // the tables, when this scratch made them
    float *h, *qkv, *ln2;           // tile buffers (h: debug runs only)
    int8_t *i8, *g8;                // tile int8 buffers (matmul input / expand output)
    uint8_t *d_ln1, *d_attn, *d_dw, *d_pre, *d_gelu;    // debug tensors being assembled
} scratch;

// Key/value storage per block: allocated separately when every block keeps its own.
static size_t kv_rows(const scratch *s) { return (size_t)s->ring * TV_KCH; }
static void kv_alloc(const scratch *s, int8_t **kh, int8_t **vt) {
    size_t vsize = (size_t)TV_HEADS * kv_rows(s) * TV_HEAD_DIM;
    *kh = tv_alloc(vsize);
    *vt = tv_alloc(vsize + 64);                                          // tv_attn_pv48 reads past a chunk
    memset(*vt, 0, vsize);                                              // keys past T stay 0
}

// rotary: tables from rotary_table(T), or NULL to make them here.
static void scratch_alloc(scratch *s, int T, int debug, int ring, const float *rotary) {
    memset(s, 0, sizeof *s);
    s->T = T;
    s->nchunks = (T + TV_KCH - 1) / TV_KCH;
    s->ring = ring > 0 && ring < s->nchunks ? ring : s->nchunks;
    s->q8 = tv_alloc(kv_rows(s) * TV_C);
    kv_alloc(s, &s->kh, &s->vt);
    for (int i = 0; i < 2; i++) s->vacc[i] = tv_alloc((size_t)TV_TILE * TV_HEAD_DIM * sizeof(int32_t));
    for (int i = 0; i < 2; i++) s->cmax[i] = tv_alloc((size_t)TV_TILE * s->nchunks * sizeof(int32_t));
    for (int i = 0; i < 2; i++) s->w8[i] = tv_alloc_fast((size_t)TV_TILE * TV_KCH);
    if (debug) s->h = tv_alloc((size_t)TV_TILE * TV_FF * sizeof(float));     // unfused expand only
    s->qkv = tv_alloc((size_t)TV_TILE * 3 * TV_C * sizeof(float));
    s->ln2 = tv_alloc((size_t)(TV_TILE + 6) * TV_C * sizeof(float));
    if (!rotary) rotary = s->own_rotary = rotary_table(T);
    s->cos_t = rotary;
    s->sin_t = rotary + (size_t)T * TV_HEAD_DIM / 2;
    s->i8 = tv_alloc_fast((size_t)TV_TILE * TV_C);                     // matmul inputs
    s->g8 = tv_alloc_fast((size_t)TV_TILE * TV_FF);
}

static void scratch_free(scratch *s) {
    tv_free(s->q8); tv_free(s->kh); tv_free(s->vt);
    for (int i = 0; i < 2; i++) { tv_free(s->vacc[i]); tv_free(s->cmax[i]); tv_free(s->w8[i]); }
    tv_free(s->h); tv_free(s->qkv); tv_free(s->ln2); tv_free(s->own_rotary);
    tv_free(s->i8); tv_free(s->g8);
}

static inline int8_t *q_row(const scratch *s, int t) { return s->q8 + (size_t)t % kv_rows(s) * TV_C; }
static inline int8_t *k_row(const scratch *s, int h, int t) {
    return s->kh + ((size_t)h * kv_rows(s) + (size_t)t % kv_rows(s)) * TV_HEAD_DIM;
}
static inline int8_t *v_chunk(const scratch *s, int h, int c) {
    return s->vt + ((size_t)h * s->ring + (size_t)c % s->ring) * TV_HEAD_DIM * TV_KCH;
}
static inline size_t vt_index(const scratch *s, int h, int t, int d) {
    return (size_t)(v_chunk(s, h, t / TV_KCH) - s->vt) + (size_t)d * TV_KCH + t % TV_KCH;
}

// Debug views of q/k/v in the golden layout [head][t][dim].
static void dbg_qkv(ctx *c, const char *prefix, const scratch *s) {
    if (!c->opt->debug) return;
    const int T = s->T;
    char name[96];
    int8_t *tmp = tv_alloc((size_t)T * TV_C);
    for (int h = 0; h < TV_HEADS; h++)
        for (int t = 0; t < T; t++)
            memcpy(tmp + ((size_t)h * T + t) * TV_HEAD_DIM, q_row(s, t) + h * TV_HEAD_DIM, TV_HEAD_DIM);
    snprintf(name, sizeof name, "%s.q", prefix); dbg(c, name, TV_I8, tmp, TV_HEADS * T, TV_HEAD_DIM);
    for (int h = 0; h < TV_HEADS; h++)
        for (int t = 0; t < T; t++) memcpy(tmp + ((size_t)h * T + t) * TV_HEAD_DIM, k_row(s, h, t), TV_HEAD_DIM);
    snprintf(name, sizeof name, "%s.k", prefix); dbg(c, name, TV_I8, tmp, TV_HEADS * T, TV_HEAD_DIM);
    for (int h = 0; h < TV_HEADS; h++)
        for (int t = 0; t < T; t++)
            for (int d = 0; d < TV_HEAD_DIM; d++)
                tmp[((size_t)h * T + t) * TV_HEAD_DIM + d] = s->vt[vt_index(s, h, t, d)];
    snprintf(name, sizeof name, "%s.v", prefix); dbg(c, name, TV_I8, tmp, TV_HEADS * T, TV_HEAD_DIM);
    tv_free(tmp);
}

// Attention for queries [q0, q0+qn) and heads [h0, h1), writing tile rows q - q0. Two passes over
// key chunks, so no score matrix is stored: pass 1 finds each query's max dot product, pass 2
// recomputes the dots, turns them into int8 weights and accumulates weights . values per chunk.
// Integer sums, so the result equals the one-pass computation exactly.
typedef struct {
    const tv_block *b;
    scratch *s;
    int q0, qn, h0, h1, set;        // queries [q0, q0+qn) are rows [0, qn) of the tile
    uint64_t t_qk, t_pv;            // time per pass on this job's core
} attn_job;

// Keys [k0 + *lo, k0 + hi) of the chunk [k0, k0 + kn) that query position q may see; returns hi.
static inline int visible(const scratch *s, int q, int k0, int kn, int *lo) {
    *lo = 0;
    if (s->lookahead <= 0) return kn;
    int hi = q + s->lookahead + 1 - k0;
    hi = hi < 0 ? 0 : hi > kn ? kn : hi;
    if (s->past > 0 && q - s->past > k0) *lo = q - s->past - k0 < hi ? q - s->past - k0 : hi;
    return hi;
}

static void attn_run(void *arg) {
    attn_job *j = arg;
    scratch *s = j->s;
    const tv_block *b = j->b;
    const int T = s->T;
    int32_t *acc = s->vacc[j->set], *cmax = s->cmax[j->set];
    const int nch = s->nchunks;
    int32_t mx[TV_TILE], sum[TV_TILE];
    uint8_t skip[TV_TILE];
    int8_t *w8 = s->w8[j->set];
    const uint8_t *lut = b->softmax_lut;
    const uint32_t lim = (uint32_t)b->softmax_n;
    for (int h = j->h0; h < j->h1; h++) {
        uint64_t t0 = tv_time_us();
        for (int r = 0; r < j->qn; r++) mx[r] = INT32_MIN;
        for (int c = 0; c < s->nchunks; c++) {
            int k0 = c * TV_KCH, kn = T - k0 < TV_KCH ? T - k0 : TV_KCH;
            for (int r = 0; r < j->qn; r++) {
                int lo, hi = visible(s, j->q0 + r, k0, kn, &lo);
                int32_t cm = hi > lo ? tv_attn_max48(q_row(s, j->q0 + r) + h * TV_HEAD_DIM, k_row(s, h, k0 + lo), hi - lo, INT32_MIN)
                                     : INT32_MIN;
                cmax[r * nch + c] = cm;
                if (cm > mx[r]) mx[r] = cm;
            }
        }
        uint64_t t1 = tv_time_us();
        memset(acc, 0, (size_t)j->qn * TV_HEAD_DIM * sizeof(int32_t));
        memset(sum, 0, sizeof sum);
        for (int c = 0; c < s->nchunks; c++) {
            int k0 = c * TV_KCH, kn = T - k0 < TV_KCH ? T - k0 : TV_KCH;
            const int8_t *vc = v_chunk(s, h, c);
            // All weights of the chunk first, then all weights . values: each step streams one
            // chunk (keys, then values) instead of alternating between the two.
            for (int r = 0; r < j->qn; r++) {
                int lo, hi = visible(s, j->q0 + r, k0, kn, &lo);
                skip[r] = hi <= lo || (uint32_t)(mx[r] - cmax[r * nch + c]) >= lim;   // every weight of the chunk is 0
                if (skip[r]) continue;
                int8_t *w = w8 + (size_t)r * TV_KCH;
                if (lo) memset(w, 0, lo);
                if (hi < TV_KCH) memset(w + hi, 0, TV_KCH - hi);
                // Row-max scaled int8 weights (largest = 127), normalized by their own sum.
                sum[r] += tv_attn_weights48(q_row(s, j->q0 + r) + h * TV_HEAD_DIM, k_row(s, h, k0 + lo), hi - lo, mx[r], lut, lim, w + lo);
            }
            for (int r = 0; r < j->qn; r++)
                if (!skip[r]) tv_attn_pv48(w8 + (size_t)r * TV_KCH, vc, acc + (size_t)r * TV_HEAD_DIM);
        }
        const float inv_attn = 1.0f / b->s_attn;
        for (int r = 0; r < j->qn; r++) {           // attention output, straight to the int8 codes of `out`
            const int32_t *ar = acc + (size_t)r * TV_HEAD_DIM;
            int8_t *yr = s->i8 + (size_t)r * TV_C + h * TV_HEAD_DIM;
            for (int d = 0; d < TV_HEAD_DIM; d++) yr[d] = tv_quant((float)ar[d] * (b->s_v / (float)sum[r]), inv_attn);
        }
        j->t_qk += t1 - t0; j->t_pv += tv_time_us() - t1;
    }
}

// Row-wise tile work. Every row is computed exactly as before; the rows are just split over the
// two cores.
enum { TILE_LN1, TILE_QKV, TILE_LN2, TILE_DW };
typedef struct {
    int op, r0, r1;                 // rows [r0, r1) of the tile
    const tv_block *b;
    scratch *s;
    const float *x;                 // residual rows of the tile
    int t0, T;
} tile_job;

static void tile_rows(void *arg) {
    const tile_job *j = arg;
    const tv_block *b = j->b;
    scratch *s = j->s;
    const int half = TV_HEAD_DIM / 2, t0 = j->t0, T = j->T;
    switch (j->op) {
    case TILE_LN1:
        layer_norm_q8_rows(j->x, j->r0, j->r1, TV_C, b->norm1_w, b->norm1_b, b->s_ln1, s->i8);
        break;
    case TILE_QKV: {                // rotary on q and k, then int8 q, k (per head) and v (transposed)
        int8_t kq[TV_C], vq[TV_C];
        const float iq = 1.0f / b->s_q, ik = 1.0f / b->s_k, iv = 1.0f / b->s_v;
        const float *cos_t = s->cos_t + (size_t)t0 * half, *sin_t = s->sin_t + (size_t)t0 * half;
        for (int r = j->r0; r < j->r1; r++) {
            const float *row = s->qkv + (size_t)r * 3 * TV_C, *cs = cos_t + r * half, *sn = sin_t + r * half;
            int t = t0 + r;
            int8_t *q = q_row(s, t);
            for (int part = 0; part < 2; part++) {
                const float *z = row + part * TV_C;
                int8_t *o = part ? kq : q;
                const float inv = part ? ik : iq;
                for (int h = 0; h < TV_HEADS; h++, z += TV_HEAD_DIM, o += TV_HEAD_DIM)
                    for (int i = 0; i < half; i++) {
                        float a = z[2 * i], bb = z[2 * i + 1];
                        o[2 * i] = tv_quant(a * cs[i] - bb * sn[i], inv);
                        o[2 * i + 1] = tv_quant(a * sn[i] + bb * cs[i], inv);
                    }
            }
            for (int i = 0; i < TV_C; i++) vq[i] = tv_quant(row[2 * TV_C + i], iv);
            for (int h = 0; h < TV_HEADS; h++) {
                memcpy(k_row(s, h, t), kq + h * TV_HEAD_DIM, TV_HEAD_DIM);
                for (int d = 0; d < TV_HEAD_DIM; d++) s->vt[vt_index(s, h, t, d)] = vq[h * TV_HEAD_DIM + d];
            }
        }
        break;
    }
    case TILE_LN2:                  // ln2 row r + 3 is position t0 + r
        layer_norm_rows(j->x, j->r0, j->r1, TV_C, b->norm2_w, b->norm2_b, s->ln2 + 3 * TV_C);
        break;
    case TILE_DW: {
        const float dw_inv = 1.0f / b->s_dw;
        for (int r = j->r0; r < j->r1; r++) {
            if (t0 + r >= TV_KERNEL / 2 && t0 + r + TV_KERNEL / 2 < T) {
                // All taps inside: no bounds checks, and 4 channels at a time so their sums
                // overlap (each still adds its taps in order).
                const float *x0 = s->ln2 + (size_t)r * TV_C;       // ln2 row r + k is tap k
                int8_t *o = s->i8 + (size_t)r * TV_C;
                for (int ch = 0; ch < TV_C; ch += 4) {
                    const float *w = b->dw_w + ch * TV_KERNEL, *x = x0 + ch;
                    float v0 = b->dw_b[ch], v1 = b->dw_b[ch + 1], v2 = b->dw_b[ch + 2], v3 = b->dw_b[ch + 3];
                    for (int k = 0; k < TV_KERNEL; k++, x += TV_C) {
                        v0 += w[k] * x[0];
                        v1 += w[TV_KERNEL + k] * x[1];
                        v2 += w[2 * TV_KERNEL + k] * x[2];
                        v3 += w[3 * TV_KERNEL + k] * x[3];
                    }
                    o[ch] = tv_quant(v0, dw_inv); o[ch + 1] = tv_quant(v1, dw_inv);
                    o[ch + 2] = tv_quant(v2, dw_inv); o[ch + 3] = tv_quant(v3, dw_inv);
                }
                continue;
            }
            for (int ch = 0; ch < TV_C; ch++) {
                float v = b->dw_b[ch];
                for (int k = 0; k < TV_KERNEL; k++) {
                    int src = t0 + r + k - TV_KERNEL / 2;
                    if (src >= 0 && src < T) v += b->dw_w[ch * TV_KERNEL + k] * s->ln2[(size_t)(src - t0 + 3) * TV_C + ch];
                }
                s->i8[(size_t)r * TV_C + ch] = tv_quant(v, dw_inv);
            }
        }
        break;
    }
    }
}

static void tile_parallel(int prof, int op, int rows, const tv_block *b, scratch *s, const float *x, int t0, int T) {
    uint64_t start = tv_time_us();
    tile_job ja = {op, 0, rows / 2, b, s, x, t0, T}, jb = {op, rows / 2, rows, b, s, x, t0, T};
    tv_parallel2(tile_rows, &ja, &jb);
    tv_prof[prof] += tv_time_us() - start;
}

// How far one block got: rows [0, p1) have Q/K/V, [0, p2) attention, [0, p3) the depthwise conv and
// FF (the block's output). carry is LayerNorm2 of rows p3-3 .. p3-1 from before their FF update.
typedef struct {
    int p1, p2, p3;
    float carry[3 * TV_C];
} block_pos;

// Advances block b over the residual x so that rows [0, target) are final, given that its input
// rows [0, avail) are final. Rows are independent of how they are grouped into tiles, so any
// sequence of calls gives the same result as one call with target = avail = T.
// prefix names the block's debug tensors.
static void block_advance(ctx *c, const tv_block *b, const char *prefix, float *x, int T, scratch *s,
                          block_pos *pos, int avail, int target) {
    const int L = s->lookahead;
    // ---- pass 1: queries, keys and values of the input rows that are final (with a look-ahead, only
    // as far as the queries below need: in a ring, keys written early would replace ones still seen)
    int n1 = L > 0 && target + 3 + L < avail ? target + 3 + L : avail, had = pos->p1;
    while (pos->p1 < n1) {
        int t0 = pos->p1, rows = n1 - t0 < TV_TILE ? n1 - t0 : TV_TILE;
        tile_parallel(TV_P_NORM, TILE_LN1, rows, b, s, x + (size_t)t0 * TV_C, t0, T);
        dbg_tile(c, &s->d_ln1, prefix, "ln1", s->i8, t0, rows, TV_C, T);
        linear_f(c, &b->qkv, s->i8, rows, s->qkv);
        tile_parallel(TV_P_ROPE, TILE_QKV, rows, b, s, NULL, t0, T);
        pos->p1 += rows;
    }
    if (had < T && pos->p1 == T) dbg_qkv(c, prefix, s);
    // ---- pass 2: attention for queries whose keys are all there (heads split over the cores)
    int n2 = pos->p1 == T ? T : L > 0 && pos->p1 > L ? pos->p1 - L : 0;
    if (n2 > target + 3) n2 = target + 3 < T ? target + 3 : T;        // the conv needs 3 rows more
    while (pos->p2 < n2) {
        int t0 = pos->p2, rows = n2 - t0 < TV_TILE ? n2 - t0 : TV_TILE;
        int oldest = s->past > 0 && t0 > s->past ? t0 - s->past : 0;        // the oldest key these queries see
        if ((pos->p1 - 1) / TV_KCH - oldest / TV_KCH >= s->ring) { fprintf(stderr, "tv: key ring too small\n"); abort(); }
        attn_job ja = {.b = b, .s = s, .q0 = t0, .qn = rows, .h0 = 0, .h1 = TV_HEADS / 2, .set = 0};
        attn_job jb = {.b = b, .s = s, .q0 = t0, .qn = rows, .h0 = TV_HEADS / 2, .h1 = TV_HEADS, .set = 1};
        tv_parallel2(attn_run, &ja, &jb);
        tv_prof[TV_P_ATTN_DOT] += ja.t_qk + jb.t_qk;     // pass 1
        tv_prof[TV_P_ATTN_PV] += ja.t_pv + jb.t_pv;      // pass 2: dots again, weights, weights . values, codes
        dbg_tile(c, &s->d_attn, prefix, "attn", s->i8, t0, rows, TV_C, T);
        linear_add(c, &b->out, s->i8, rows, x + (size_t)t0 * TV_C);
        pos->p2 += rows;
    }
    // ---- pass 3: depthwise conv over LayerNorm2 (3 rows of context), expand, GELU, project
    int n3 = pos->p2 == T ? T : pos->p2 > 3 ? pos->p2 - 3 : 0;
    if (n3 > target) n3 = target;
    while (pos->p3 < n3) {
        int t0 = pos->p3, rows = n3 - t0 < TV_TILE ? n3 - t0 : TV_TILE;
        int last = t0 + rows + 3 < T ? t0 + rows + 3 : T;            // LayerNorm rows [t0, last) are fresh
        // ln2 row r stands for position t0 - 3 + r
        if (t0 > 0) memcpy(s->ln2, pos->carry, sizeof pos->carry);
        tile_parallel(TV_P_NORM, TILE_LN2, last - t0, b, s, x + (size_t)t0 * TV_C, t0, T);
        if (t0 + rows < T) memcpy(pos->carry, s->ln2 + (size_t)rows * TV_C, sizeof pos->carry);  // positions t0+rows-3 ..
        tile_parallel(TV_P_DW, TILE_DW, rows, b, s, NULL, t0, T);
        dbg_tile(c, &s->d_dw, prefix, "dw", s->i8, t0, rows, TV_C, T);
        int8_t *g = s->g8;
        if (c->opt->debug) {        // unfused, so the pre-activation codes can be checked too
            linear_f(c, &b->expand, s->i8, rows, s->h);
            quantize(s->h, rows * TV_FF, b->s_pre, g);
            dbg_tile(c, &s->d_pre, prefix, "pre", g, t0, rows, TV_FF, T);
            TV_PROF(TV_P_GELU, for (int i = 0; i < rows * TV_FF; i++) g[i] = b->gelu_lut[g[i] + 128]);
        } else {
            linear_q8(c, &b->expand, s->i8, rows, g, b->s_pre, b->gelu_lut);
        }
        dbg_tile(c, &s->d_gelu, prefix, "gelu", g, t0, rows, TV_FF, T);
        linear_add(c, &b->project, g, rows, x + (size_t)t0 * TV_C);
        pos->p3 += rows;
    }
}

// x [T][C] is the residual input (overwritten); out receives the final LayerNorm (may alias x).
// lookahead/past: the attention mask (0: full), see tv_options.acoustic_lookahead. rotary: see scratch_alloc.
static void stack_forward(ctx *c, const tv_stack *st, const char *name, float *x, int T, float *out, int lookahead, int past,
                          const float *rotary) {
    scratch s;
    scratch_alloc(&s, T, c->opt->debug != NULL, 0, rotary);
    s.lookahead = lookahead;
    s.past = past;
    char buf[96];
    for (int i = 0; i < st->count; i++) {
        snprintf(buf, sizeof buf, "%s.blocks.%d", name, i);
        block_pos pos = {0};
        block_advance(c, &st->blocks[i], buf, x, T, &s, &pos, T, T);
        snprintf(buf, sizeof buf, "%s.%d.out", name, i);
        dbg(c, buf, TV_F32, x, T, TV_C);
    }
    scratch_free(&s);
    layer_norm(x, T, TV_C, st->norm_w, st->norm_b, out);
    snprintf(buf, sizeof buf, "%s.final", name);
    dbg(c, buf, TV_F32, out, T, TV_C);
}

static void predict_durations(ctx *c, const tv_model *m, const float *hidden, const int32_t *ids, int N, float speed,
                              int32_t *dur) {
    float *win = tv_alloc((size_t)N * 3 * TV_C * sizeof(float));
    float *h1 = tv_alloc((size_t)N * TV_PRED * sizeof(float)), *h2 = tv_alloc((size_t)N * TV_PRED * sizeof(float));
    int8_t *q = tv_alloc((size_t)N * 3 * TV_C);
    float *out = tv_alloc((size_t)N * 2 * sizeof(float));
    windows3(hidden, N, TV_C, win);
    quantize(win, N * 3 * TV_C, m->dur1.in_scale, q);
    dbg(c, "dur.in", TV_I8, q, N, 3 * TV_C);
    linear_f(c, &m->dur1, q, N, h1);
    for (int i = 0; i < N * TV_PRED; i++) h1[i] = gelu(h1[i]);
    layer_norm(h1, N, TV_PRED, m->dnorm1_w, m->dnorm1_b, h2);
    windows3(h2, N, TV_PRED, win);
    quantize(win, N * 3 * TV_PRED, m->dur2.in_scale, q);
    dbg(c, "dur.h1", TV_I8, q, N, 3 * TV_PRED);
    linear_f(c, &m->dur2, q, N, h1);
    for (int i = 0; i < N * TV_PRED; i++) h1[i] = gelu(h1[i]);
    layer_norm(h1, N, TV_PRED, m->dnorm2_w, m->dnorm2_b, h2);
    const float limit = log1pf(400.0f);
    for (int t = 0; t < N; t++) {
        for (int o = 0; o < 2; o++) {
            float v = m->dur_out_b[o];
            for (int i = 0; i < TV_PRED; i++) v += m->dur_out_w[o * TV_PRED + i] * h2[(size_t)t * TV_PRED + i];
            out[t * 2 + o] = v;
        }
        float lf = out[t * 2] < 0 ? 0 : out[t * 2] > limit ? limit : out[t * 2];
        int32_t frames = (int32_t)rintf(expm1f(lf) / speed);
        if (frames < 1) frames = 1;
        int phone = ids[t] == VOCAB_UNK || ids[t] >= FIRST_PHONE;
        dur[t] = (phone || out[t * 2 + 1] > 0) ? frames : 0;
    }
    dbg(c, "dur.out", TV_F32, out, N, 2);
    tv_free(win); tv_free(h1); tv_free(h2); tv_free(q); tv_free(out);
}

static void collect_audio(void *user, const float *x, int n) {
    float **cursor = user;
    memcpy(*cursor, x, n * sizeof(float));
    *cursor += n;
}

// Vocoder inputs of `rows` frames from the final acoustic LayerNorm and the prosody: the head input
// codes q, the harmonic and noise parameters and f0 in Hz.
static void vocoder_inputs(ctx *c, const tv_model *m, const float *ac, const float *pros, int rows, int8_t *q,
                           float *harm, float *nz, float *f0) {
    quantize(ac, rows * TV_C, m->s_voc, q);
    linear_f(c, &m->harmonic, q, rows, harm);
    linear_f(c, &m->noise, q, rows, nz);
    for (int f = 0; f < rows; f++) {
        float hz = 200.0f * expf(pros[f * 3]);
        f0[f] = hz < 50.0f ? 50.0f : hz > 600.0f ? 600.0f : hz;
    }
}

long tv_vocode(const tv_model *m, const float *ac, const float *pros, int T, const tv_options *opt, float *audio) {
    ctx c = {opt, 0};
    int8_t *q = tv_alloc((size_t)T * TV_C);
    float *harm = tv_alloc((size_t)T * TV_HARM_PARAMS * sizeof(float)), *nz = tv_alloc((size_t)T * TV_NOISE_PARAMS * sizeof(float));
    float *f0 = tv_alloc((size_t)T * sizeof(float));
    vocoder_inputs(&c, m, ac, pros, T, q, harm, nz, f0);
    dbg(&c, "voc.in", TV_I8, q, T, TV_C);
    int S = T * TV_HOP;
    if (opt->on_audio) {
        tv_vocoder_stream(harm, nz, f0, opt->noise, opt->seed, T, 0, opt->on_audio, opt->audio_user, NULL, NULL);
    } else {
        float *cursor = audio;
        tv_vocoder_stream(harm, nz, f0, opt->noise, opt->seed, T, 0, collect_audio, &cursor, NULL, NULL);
        dbg(&c, "audio", TV_F32, audio, 1, S);
    }
    tv_free(q); tv_free(harm); tv_free(nz); tv_free(f0);
    return c.saturations;
}

// ---------------------------------------------------------------- streaming acoustic stack
// The acoustic stack runs a little ahead of the vocoder: before each vocoder block, every block
// advances just far enough. With a look-ahead L, output row t of a block needs its input rows up
// to t + L + 3 (the conv's 3 rows), and with a past window it attends to the last `past` rows
// only, so keys, values, queries and the vocoder inputs live in rings that do not grow with the
// utterance. Look-ahead 0 means full attention: every block runs over the whole utterance first,
// exactly as without streaming. The rows are computed exactly as in one pass over the utterance.
#define VOC_RING 256                // frames of vocoder input kept (it reads within ~110 of its position)
typedef struct {
    ctx *c;
    const tv_model *m;
    int T, D;                       // D: input rows a block needs beyond its output rows
    float *x;                       // residual [T][C] (overwritten)
    const float *pros;
    scratch s;                      // shared tile buffers and queries; kh/vt point at the current block's
    int8_t **kh, **vt;              // per block
    block_pos *pos;                 // per block
    int vring;                      // rows of harm, nz and f0 (a ring of VOC_RING, or T)
    float *harm, *nz, *f0;          // vocoder inputs [vring][TV_HARM_PARAMS], [vring][TV_NOISE_PARAMS], [vring]
    float *ac_t, *harm_t, *nz_t, *f0_t;     // one step (at most TV_TILE rows) before it goes into the rings
    int8_t *q_t;
    int done;                       // rows of the vocoder inputs that are final
    char prefix[TV_ACOUSTIC_BLOCKS][40];    // debug names of the blocks
} acoustic_stream;

// Makes stack output rows [done, e) final (at most TV_TILE of them) and turns them into vocoder inputs.
static void acoustic_step(acoustic_stream *a, int e) {
    const tv_stack *st = &a->m->acoustic;
    const int T = a->T, n = st->count;
    if (a->s.ring < a->s.nchunks) {     // block 0 writes queries up to here; no block may still need the old ones
        long newest = e + (long)(n - 1) * a->D + a->s.lookahead + TV_KERNEL / 2;
        int oldest = a->pos[n - 1].p2;
        for (int l = 0; l < n; l++) if (a->pos[l].p2 < oldest) oldest = a->pos[l].p2;
        if ((newest < T ? newest : T) - oldest > (long)kv_rows(&a->s)) { fprintf(stderr, "tv: query ring too small\n"); abort(); }
    }
    for (int l = 0; l < n; l++) {
        long target = e + (long)(n - 1 - l) * a->D;
        a->s.kh = a->kh[l]; a->s.vt = a->vt[l];
        block_advance(a->c, &st->blocks[l], a->prefix[l], a->x, T, &a->s, &a->pos[l], l ? a->pos[l - 1].p3 : T,
                      target < T ? (int)target : T);
    }
    if (a->pos[n - 1].p3 < e) { fprintf(stderr, "tv: acoustic stream stalled at %d of %d\n", a->pos[n - 1].p3, e); abort(); }
    // final LayerNorm and vocoder inputs of the new rows, copied into the rings
    const int r0 = a->done, rows = e - r0;
    TV_PROF(TV_P_NORM, layer_norm_rows(a->x + (size_t)r0 * TV_C, 0, rows, TV_C, st->norm_w, st->norm_b, a->ac_t));
    vocoder_inputs(a->c, a->m, a->ac_t, a->pros + (size_t)r0 * 3, rows, a->q_t, a->harm_t, a->nz_t, a->f0_t);
    for (int r = 0; r < rows; r++) {
        int slot = (r0 + r) % a->vring;
        memcpy(a->harm + (size_t)slot * TV_HARM_PARAMS, a->harm_t + (size_t)r * TV_HARM_PARAMS, TV_HARM_PARAMS * sizeof(float));
        memcpy(a->nz + (size_t)slot * TV_NOISE_PARAMS, a->nz_t + (size_t)r * TV_NOISE_PARAMS, TV_NOISE_PARAMS * sizeof(float));
        a->f0[slot] = a->f0_t[r];
    }
    a->done = e;
}

static void acoustic_advance(void *user, int frames) {
    acoustic_stream *a = user;
    if (frames <= a->done) return;
    // after the first rows (as few as the vocoder needs: sound starts sooner), whole tiles: faster matmuls
    int E = a->done && frames < a->done + TV_TILE ? a->done + TV_TILE : frames;
    if (E > a->T) E = a->T;
    // the vocoder still reads frames from about frames - 48 on (a block of tv_dsp.c's VOC_BLOCK = 8192
    // samples is 32 frames, plus its FFT windows and a margin): they must not be overwritten
    if (a->vring < a->T && E - (frames - 48) > a->vring) { fprintf(stderr, "tv: vocoder ring too small\n"); abort(); }
    while (a->done < E) acoustic_step(a, E - a->done > TV_TILE ? a->done + TV_TILE : E);
}

static void acoustic_stream_run(ctx *c, const tv_model *m, float *x, const float *pros, int T, int L, int P,
                                const float *rotary, float *audio) {
    const tv_options *opt = c->opt;
    acoustic_stream a = {.c = c, .m = m, .T = T, .D = L ? L + TV_KERNEL / 2 : T, .x = x, .pros = pros};
    const int n = m->acoustic.count;
    for (int l = 0; l < n; l++) snprintf(a.prefix[l], sizeof a.prefix[l], "acoustic_stack.blocks.%d", l);
    // Rows the rings must hold: a block's keys from its oldest query - P to its newest + L + 3 plus
    // one step, and the queries of all blocks at once (block 0 runs (n-1) * D rows ahead of the last).
    long keys = (long)P + L + TV_KERNEL / 2 + TV_TILE, queries = (long)(n - 1) * a.D + L + TV_KERNEL / 2 + 2 * TV_TILE;
    long rows = keys > queries ? keys : queries;
    int ring = P > 0 && rows < T ? (int)((rows + TV_KCH - 1) / TV_KCH) + 1 : 0;
    scratch_alloc(&a.s, T, 0, ring, rotary);
    a.s.lookahead = L;
    a.s.past = P;
    a.kh = tv_alloc(n * sizeof *a.kh);
    a.vt = tv_alloc(n * sizeof *a.vt);
    a.pos = tv_alloc(n * sizeof *a.pos);
    memset(a.pos, 0, n * sizeof *a.pos);
    for (int l = 0; l < n; l++) {         // without a look-ahead the blocks run one after the other: one set
        if (l && L) kv_alloc(&a.s, &a.kh[l], &a.vt[l]);
        else { a.kh[l] = a.s.kh; a.vt[l] = a.s.vt; }
    }
    a.vring = L && T > VOC_RING ? VOC_RING : T;
    a.harm = tv_alloc((size_t)a.vring * TV_HARM_PARAMS * sizeof(float));
    a.nz = tv_alloc((size_t)a.vring * TV_NOISE_PARAMS * sizeof(float));
    a.f0 = tv_alloc((size_t)a.vring * sizeof(float));
    a.ac_t = tv_alloc((size_t)TV_TILE * TV_C * sizeof(float));
    a.harm_t = tv_alloc((size_t)TV_TILE * TV_HARM_PARAMS * sizeof(float));
    a.nz_t = tv_alloc((size_t)TV_TILE * TV_NOISE_PARAMS * sizeof(float));
    a.f0_t = tv_alloc((size_t)TV_TILE * sizeof(float));
    a.q_t = tv_alloc((size_t)TV_TILE * TV_C);
    float *cursor = audio;
    tv_vocoder_stream(a.harm, a.nz, a.f0, opt->noise, opt->seed, T, a.vring, opt->on_audio ? opt->on_audio : collect_audio,
                      opt->on_audio ? opt->audio_user : (void *)&cursor, acoustic_advance, &a);
    for (int l = 1; l < n && L; l++) { tv_free(a.kh[l]); tv_free(a.vt[l]); }
    a.s.kh = a.kh[0]; a.s.vt = a.vt[0];
    scratch_free(&a.s);
    tv_free(a.kh); tv_free(a.vt); tv_free(a.pos);
    tv_free(a.harm); tv_free(a.nz); tv_free(a.f0);
    tv_free(a.ac_t); tv_free(a.harm_t); tv_free(a.nz_t); tv_free(a.f0_t); tv_free(a.q_t);
}

int tv_synthesize(const tv_model *m, const int32_t *ids, const int32_t *stress, int N, int voice,
                  const tv_options *opt, tv_result *res) {
    ctx c = {opt, 0};
    memset(res, 0, sizeof *res);
    if (N < 1 || voice < 0 || voice >= TV_VOICES) return TV_BAD_INPUT;
    for (int t = 0; t < N; t++)
        if (ids[t] < 0 || ids[t] >= TV_VOCAB || stress[t] < 0 || stress[t] >= TV_STRESS) return TV_BAD_INPUT;
    // ---- text encoder (its final LayerNorm replaces x)
    float *x = tv_alloc((size_t)N * TV_C * sizeof(float));
    for (int t = 0; t < N; t++)
        for (int i = 0; i < TV_C; i++)
            x[t * TV_C + i] = m->tokens[ids[t] * TV_C + i] + m->stress[stress[t] * TV_C + i] + m->voices[voice * TV_C + i];
    stack_forward(&c, &m->encoder, "encoder", x, N, x, 0, 0, NULL);
    const float *hidden = x;
    // ---- durations
    int32_t *dur = tv_alloc((size_t)N * sizeof(int32_t));
    predict_durations(&c, m, hidden, ids, N, opt->speed > 0 ? opt->speed : 1.0f, dur);
    if (opt->force_durations) memcpy(dur, opt->force_durations, N * sizeof(int32_t));
    dbg(&c, "durations", TV_I32, dur, 1, N);
    long frames_total = 0;
    int rc = 0;
    for (int t = 0; t < N && !rc; t++) {
        if (dur[t] < 0) rc = TV_BAD_INPUT;
        frames_total += dur[t];
        if (frames_total > INT32_MAX / TV_HOP) rc = TV_TOO_LONG;
    }
    const int T = rc ? 0 : (int)frames_total;
    if (!rc && T < 1) rc = TV_NO_FRAMES;
    if (!rc && opt->max_frames > 0 && T > opt->max_frames) rc = TV_TOO_LONG;
    if (!rc && opt->noise && opt->noise_len < T * TV_HOP) rc = TV_BAD_INPUT;
    if (rc) {
        if (rc == TV_TOO_LONG) res->n_frames = frames_total > INT32_MAX ? INT32_MAX : (int)frames_total;
        tv_free(x); tv_free(dur);
        return rc;
    }
    // ---- repeat token states by duration, add position-in-token and voice
    float *frames = tv_alloc((size_t)T * TV_C * sizeof(float));
    uint8_t *vowel = tv_alloc(T);
    for (int t = 0, f = 0; t < N; t++) {
        int owned = dur[t] < 1 ? 1 : dur[t];
        float feat[3] = {0, 0, log1pf((float)owned) / 4.0f};
        for (int o = 0; o < dur[t]; o++, f++) {
            feat[0] = ((float)o + 0.5f) / (float)owned;
            feat[1] = 1.0f - feat[0];
            vowel[f] = ids[t] >= FIRST_VOWEL;
            for (int i = 0; i < TV_C; i++) {
                const float *w = m->frame_pos_w + i * 3;
                frames[(size_t)f * TV_C + i] = hidden[t * TV_C + i] +
                    (w[0] * feat[0] + w[1] * feat[1] + w[2] * feat[2] + m->frame_pos_b[i]) + m->frame_voices[voice * TV_C + i];
            }
        }
    }
    tv_free(x);
    // ---- prosody (its output replaces the frames; each row is read whole before it is overwritten)
    float *rotary = rotary_table(T);    // the prosody and acoustic stacks run over the same positions
    stack_forward(&c, &m->prosody, "prosody_stack", frames, T, frames, 0, 0, rotary);
    float *pros = tv_alloc((size_t)T * 3 * sizeof(float));
    float *ac_in = frames;
    for (int f = 0; f < T; f++) {
        const float *row = frames + (size_t)f * TV_C;
        for (int o = 0; o < 3; o++) {
            float v = m->prosody_head_b[o];
            for (int i = 0; i < TV_C; i++) v += m->prosody_head_w[o * TV_C + i] * row[i];
            pros[f * 3 + o] = v;
        }
        if (vowel[f] && pros[f * 3 + 1] < 2.0f) pros[f * 3 + 1] = 2.0f;
        float p0 = pros[f * 3], p2 = pros[f * 3 + 2];
        float cond[3] = {p0 < -2.0f ? -2.0f : p0 > 1.5f ? 1.5f : p0, pros[f * 3 + 1] > 0 ? 1.0f : 0.0f,
                         p2 < -4.5f ? -4.5f : p2 > 3.0f ? 3.0f : p2};
        for (int i = 0; i < TV_C; i++) {
            const float *w = m->prosody_proj_w + i * 3;
            ac_in[(size_t)f * TV_C + i] = row[i] + (w[0] * cond[0] + w[1] * cond[1] + w[2] * cond[2] + m->prosody_proj_b[i]);
        }
    }
    tv_free(vowel);
    dbg(&c, "prosody", TV_F32, pros, T, 3);
    // ---- acoustic stack, vocoder heads (int8) and source-filter synthesis (float)
    int S = T * TV_HOP;
    res->audio = opt->on_audio ? NULL : tv_alloc((size_t)S * sizeof(float));
    const int L = opt->acoustic_lookahead > 0 ? opt->acoustic_lookahead : 0, P = L ? opt->acoustic_past : 0;
    if (!opt->debug) {
        acoustic_stream_run(&c, m, ac_in, pros, T, L, P, rotary, res->audio);     // computed as the vocoder needs it
    } else {        // reference: one pass with the same mask, every tensor whole for the debug dumps
        stack_forward(&c, &m->acoustic, "acoustic_stack", ac_in, T, ac_in, L, P, rotary);  // the final LN replaces ac_in
        dbg(&c, "acoustic", TV_F32, ac_in, T, TV_C);
        c.saturations += tv_vocode(m, ac_in, pros, T, opt, res->audio);
    }
    res->n_samples = S;
    res->n_frames = T;
    res->durations = dur;
    res->qacc_saturations = c.saturations;
    tv_free(rotary); tv_free(frames); tv_free(pros);
    return 0;
}
