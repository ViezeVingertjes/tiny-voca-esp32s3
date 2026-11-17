// tiny-voca engine: model loading and the forward pass. Numerics follow
// esp32/tools/tvq.py (the golden model); see that file for the specification.
#include "tv.h"
#include "tv_kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GELU_MIN (-0.16997124f)
#define VOCAB_UNK 4
#define FIRST_PHONE 17
#define FIRST_VOWEL 43

// ---------------------------------------------------------------- container
int tv_file_open(tv_file *f, const void *data, size_t size) {
    const uint8_t *p = data;
    if (size < 16 || memcmp(p, "TVQ1", 4)) return -1;
    uint32_t version, count;
    memcpy(&version, p + 4, 4);
    memcpy(&count, p + 8, 4);
    if (version != 1 || 16 + (size_t)count * sizeof(tv_entry) > size) return -1;
    f->base = p;
    f->entries = (const tv_entry *)(p + 16);
    f->count = count;
    for (uint32_t i = 0; i < count; i++)
        if ((size_t)f->entries[i].offset + f->entries[i].nbytes > size) return -1;
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

static const void *get(loader *L, const char *name, uint32_t dtype, uint32_t nelem) {
    const void *p = tv_file_get(&L->m->file, name, dtype, nelem);
    if (!p && !L->failed) {
        snprintf(L->m->error, sizeof L->m->error, "missing or malformed tensor %s (%u elements)", name, (unsigned)nelem);
        L->failed = 1;
    }
    return p;
}

static const float *getf(loader *L, const char *fmt, const char *prefix, uint32_t nelem) {
    char name[96];
    snprintf(name, sizeof name, fmt, prefix);
    return get(L, name, TV_F32, nelem);
}

static float scale_of(loader *L, const char *prefix, const char *point) {
    char name[96];
    snprintf(name, sizeof name, "%s.%s.max", prefix, point);
    const float *v = get(L, name, TV_F32, 1);
    return v ? (float)((double)*v / 127.0) : 1.0f;
}

static void load_linear(loader *L, tv_linear *lin, const char *base, int N, int K, float in_scale, int chunks) {
    char name[96];
    int npad = (N + 15) / 16 * 16;
    lin->N = N, lin->K = K, lin->in_scale = in_scale, lin->chunks = 1, lin->pre = NULL;
    snprintf(name, sizeof name, "%s.wq", base); lin->wq = get(L, name, TV_I8, npad * K);
    snprintf(name, sizeof name, "%s.ws", base); lin->ws = get(L, name, TV_F32, N);
    snprintf(name, sizeof name, "%s.b", base); lin->b = get(L, name, TV_F32, N);
    if (chunks > 1) {
        snprintf(name, sizeof name, "%s.pre", base);
        lin->pre = get(L, name, TV_I32, chunks * N);
        lin->chunks = chunks;
    }
    lin->bias = calloc(N, sizeof(float));
    if (lin->b) memcpy(lin->bias, lin->b, N * sizeof(float));
    if (!L->failed && tv_linear_prepare(lin)) {
        snprintf(L->m->error, sizeof L->m->error, "kernel preparation failed for %s", base);
        L->failed = 1;
    }
}

static float gelu(float x) { return 0.5f * x * (1.0f + erff(x * (float)M_SQRT1_2)); }

static void load_block(loader *L, tv_block *b, const char *prefix) {
    char base[96];
    b->norm1_w = getf(L, "%s.norm1.weight", prefix, TV_C);
    b->norm1_b = getf(L, "%s.norm1.bias", prefix, TV_C);
    b->norm2_w = getf(L, "%s.norm2.weight", prefix, TV_C);
    b->norm2_b = getf(L, "%s.norm2.bias", prefix, TV_C);
    b->dw_w = getf(L, "%s.depthwise.weight", prefix, TV_C * TV_KERNEL);
    b->dw_b = getf(L, "%s.depthwise.bias", prefix, TV_C);
    b->s_ln1 = scale_of(L, prefix, "ln1");
    b->s_q = scale_of(L, prefix, "q");
    b->s_k = scale_of(L, prefix, "k");
    b->s_v = scale_of(L, prefix, "v");
    b->s_attn = scale_of(L, prefix, "attn");
    b->s_dw = scale_of(L, prefix, "dw");
    b->s_pre = scale_of(L, prefix, "pre");
    float gmax = scale_of(L, prefix, "gelu") * 127.0f;     // undo the /127: the stored max
    char name[96];
    snprintf(name, sizeof name, "%s.gelu.max", prefix);
    const float *gm = tv_file_get(&L->m->file, name, TV_F32, 1);
    if (gm) gmax = *gm;
    b->s_gelu = (float)(((double)gmax - (double)GELU_MIN) / 255.0);
    snprintf(base, sizeof base, "%s.attention.qkv", prefix); load_linear(L, &b->qkv, base, 3 * TV_C, TV_C, b->s_ln1, 1);
    snprintf(base, sizeof base, "%s.attention.out", prefix); load_linear(L, &b->out, base, TV_C, TV_C, b->s_attn, 1);
    snprintf(base, sizeof base, "%s.expand", prefix); load_linear(L, &b->expand, base, TV_FF, TV_C, b->s_dw, 1);
    snprintf(base, sizeof base, "%s.project", prefix); load_linear(L, &b->project, base, TV_C, TV_FF, b->s_gelu, 2);
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
    b->softmax_lut = malloc(b->softmax_n);
    for (int g = 0; g < b->softmax_n; g++) b->softmax_lut[g] = (uint8_t)lrint(127.0 * exp(-g * score_scale));
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
    char prefix[96];
    s->blocks = storage;
    s->count = count;
    for (int i = 0; i < count; i++) {
        snprintf(prefix, sizeof prefix, "%s.blocks.%d", name, i);
        load_block(L, &storage[i], prefix);
    }
    s->norm_w = getf(L, "%s.norm.weight", name, TV_C);
    s->norm_b = getf(L, "%s.norm.bias", name, TV_C);
}

int tv_model_init(tv_model *m, const void *data, size_t size) {
    memset(m, 0, sizeof *m);
    tv_kernels_init();
    if (tv_file_open(&m->file, data, size)) {
        snprintf(m->error, sizeof m->error, "not a TVQ1 file");
        return -1;
    }
    loader L = {m, 0};
    m->tokens = get(&L, "tokens.weight", TV_F32, 76 * TV_C);
    m->stress = get(&L, "stress.weight", TV_F32, 3 * TV_C);
    m->voices = get(&L, "voices.weight", TV_F32, 2 * TV_C);
    m->frame_voices = get(&L, "frame_voices.weight", TV_F32, 2 * TV_C);
    load_stack(&L, &m->encoder, m->block_storage, "encoder", 4);
    load_stack(&L, &m->prosody, m->block_storage + 4, "prosody_stack", 2);
    load_stack(&L, &m->acoustic, m->block_storage + 6, "acoustic_stack", 5);
    load_linear(&L, &m->dur1, "duration.conv1", TV_PRED, 3 * TV_C, scale_of(&L, "dur", "in"), 1);
    load_linear(&L, &m->dur2, "duration.conv2", TV_PRED, 3 * TV_PRED, scale_of(&L, "dur", "h1"), 1);
    m->dnorm1_w = get(&L, "duration.norm1.weight", TV_F32, TV_PRED);
    m->dnorm1_b = get(&L, "duration.norm1.bias", TV_F32, TV_PRED);
    m->dnorm2_w = get(&L, "duration.norm2.weight", TV_F32, TV_PRED);
    m->dnorm2_b = get(&L, "duration.norm2.bias", TV_F32, TV_PRED);
    m->dur_out_w = get(&L, "duration.out.weight", TV_F32, 2 * TV_PRED);
    m->dur_out_b = get(&L, "duration.out.bias", TV_F32, 2);
    m->frame_pos_w = get(&L, "frame_position.weight", TV_F32, TV_C * 3);
    m->frame_pos_b = get(&L, "frame_position.bias", TV_F32, TV_C);
    m->prosody_head_w = get(&L, "prosody_head.weight", TV_F32, 3 * TV_C);
    m->prosody_head_b = get(&L, "prosody_head.bias", TV_F32, 3);
    m->prosody_proj_w = get(&L, "prosody_projection.weight", TV_F32, TV_C * 3);
    m->prosody_proj_b = get(&L, "prosody_projection.bias", TV_F32, TV_C);
    m->s_voc = scale_of(&L, "voc", "in");
    load_linear(&L, &m->harmonic, "vocoder.harmonic", 98, TV_C, m->s_voc, 1);
    load_linear(&L, &m->noise, "vocoder.noise", 36, TV_C, m->s_voc, 1);
    return L.failed ? -1 : 0;
}

static void free_linear(tv_linear *l) { tv_linear_release(l); free(l->bias); l->bias = NULL; }

void tv_model_free(tv_model *m) {
    for (int i = 0; i < 11; i++) {
        tv_block *b = &m->block_storage[i];
        free_linear(&b->qkv); free_linear(&b->out); free_linear(&b->expand); free_linear(&b->project);
        free(b->softmax_lut); b->softmax_lut = NULL;
    }
    free_linear(&m->dur1); free_linear(&m->dur2); free_linear(&m->harmonic); free_linear(&m->noise);
}

// ---------------------------------------------------------------- profiling
uint64_t tv_prof[TV_P_COUNT];
static const char *const prof_names[TV_P_COUNT] = {"matmul (int8)", "matmul epilogue", "attention dots", "softmax",
    "layernorm", "depthwise conv", "quantize", "gelu table", "rotary", "vocoder cepstrum", "vocoder stft filter",
    "vocoder pulses"};
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

#define xmalloc tv_alloc

// Round half to even via the 1.5 * 2^23 trick (default rounding mode), after clamping.
static inline int8_t quant_inv(float x, float inv) {
    float q = x * inv;
    q = q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q;
    return (int8_t)((int32_t)(q + 12582912.0f) - 12582912);
}

static void quantize(const float *x, int n, float scale, int8_t *q) {
    uint64_t t0 = tv_time_us();
    const float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) q[i] = quant_inv(x[i], inv);
    tv_prof[TV_P_QUANT] += tv_time_us() - t0;
}

// y = dequantized matmul + bias (float output).
static void linear_f(ctx *c, const tv_linear *L, const int8_t *x, int T, int32_t *acc, float *y) {
    TV_PROF(TV_P_MATMUL, c->saturations += tv_matmul_i8(L, x, T, acc));
    uint64_t t0 = tv_time_us();
    for (int t = 0; t < T; t++)
        for (int n = 0; n < L->N; n++)
            y[(size_t)t * L->N + n] = (float)acc[(size_t)t * L->N + n] * (L->in_scale * L->ws[n]) + L->bias[n];
    tv_prof[TV_P_EPILOGUE] += tv_time_us() - t0;
}

static void layer_norm(const float *x, int T, int C, const float *w, const float *b, float *y) {
    uint64_t t0 = tv_time_us();
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * C;
        float *yt = y + (size_t)t * C;
        float mean = 0, var = 0;
        for (int i = 0; i < C; i++) mean += xt[i];
        mean /= C;
        for (int i = 0; i < C; i++) var += (xt[i] - mean) * (xt[i] - mean);
        float inv = 1.0f / sqrtf(var / C + 1e-5f);
        for (int i = 0; i < C; i++) yt[i] = (xt[i] - mean) * inv * w[i] + b[i];
    }
    tv_prof[TV_P_NORM] += tv_time_us() - t0;
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

static void rotary(int T, float *cos_t, float *sin_t) {
    const int half = TV_HEAD_DIM / 2;
    for (int i = 0; i < half; i++) {
        float inv = 1.0f / powf(10000.0f, (float)(2 * i) / (float)TV_HEAD_DIM);
        for (int t = 0; t < T; t++) {
            float a = (float)t * inv;
            cos_t[t * half + i] = cosf(a);
            sin_t[t * half + i] = sinf(a);
        }
    }
}

// Scratch buffers for one stack pass over T positions.
typedef struct {
    float *x, *h, *qkv, *y, *cos_t, *sin_t, *scores;
    int8_t *q8, *k8, *v8, *i8, *kh, *vt, *w8;
    int32_t *acc, *wts;
    int tpad;                   // T rounded up to 16 (V^T row length)
} scratch;

static void scratch_alloc(scratch *s, int T) {
    s->h = xmalloc((size_t)T * TV_FF * sizeof(float));
    s->qkv = xmalloc((size_t)T * 3 * TV_C * sizeof(float));
    s->y = xmalloc((size_t)T * TV_C * sizeof(float));
    s->cos_t = xmalloc((size_t)T * TV_HEAD_DIM / 2 * sizeof(float));
    s->sin_t = xmalloc((size_t)T * TV_HEAD_DIM / 2 * sizeof(float));
    s->scores = xmalloc((size_t)T * sizeof(float));
    s->wts = xmalloc((size_t)T * sizeof(int32_t));
    s->q8 = xmalloc((size_t)T * TV_C);
    s->k8 = xmalloc((size_t)T * TV_C);
    s->v8 = xmalloc((size_t)T * TV_C);
    s->i8 = xmalloc((size_t)T * TV_FF);
    s->acc = xmalloc((size_t)T * TV_FF * sizeof(int32_t));
    s->tpad = (T + 15) / 16 * 16;
    s->kh = xmalloc((size_t)T * TV_HEAD_DIM);
    s->vt = xmalloc((size_t)TV_HEAD_DIM * s->tpad);
    s->w8 = xmalloc((size_t)s->tpad);
}

static void scratch_free(scratch *s) {
    tv_free(s->h); tv_free(s->qkv); tv_free(s->y); tv_free(s->cos_t); tv_free(s->sin_t); tv_free(s->scores); tv_free(s->wts);
    tv_free(s->q8); tv_free(s->k8); tv_free(s->v8); tv_free(s->i8); tv_free(s->acc);
    tv_free(s->kh); tv_free(s->vt); tv_free(s->w8);
}

// Debug view of q/k/v in the golden layout [head][t][dim].
static void dbg_heads(ctx *c, const char *name, const int8_t *x, int T) {
    if (!c->opt->debug) return;
    int8_t *tmp = xmalloc((size_t)T * TV_C);
    for (int h = 0; h < TV_HEADS; h++)
        for (int t = 0; t < T; t++)
            memcpy(tmp + ((size_t)h * T + t) * TV_HEAD_DIM, x + (size_t)t * TV_C + h * TV_HEAD_DIM, TV_HEAD_DIM);
    dbg(c, name, TV_I8, tmp, TV_HEADS * T, TV_HEAD_DIM);
    tv_free(tmp);
}

static void attention(ctx *c, const tv_block *b, const char *prefix, const float *xn, int T, scratch *s, float *out) {
    char name[96];
    quantize(xn, T * TV_C, b->s_ln1, s->i8);
    snprintf(name, sizeof name, "%s.ln1", prefix); dbg(c, name, TV_I8, s->i8, T, TV_C);
    linear_f(c, &b->qkv, s->i8, T, s->acc, s->qkv);
    const int half = TV_HEAD_DIM / 2;
    uint64_t t_rope = tv_time_us();
    for (int t = 0; t < T; t++) {
        float *row = s->qkv + (size_t)t * 3 * TV_C;
        for (int part = 0; part < 2; part++)            // rotary on q and k
            for (int h = 0; h < TV_HEADS; h++) {
                float *z = row + part * TV_C + h * TV_HEAD_DIM;
                for (int i = 0; i < half; i++) {
                    float a = z[2 * i], bb = z[2 * i + 1], cs = s->cos_t[t * half + i], sn = s->sin_t[t * half + i];
                    z[2 * i] = a * cs - bb * sn;
                    z[2 * i + 1] = a * sn + bb * cs;
                }
            }
        tv_prof[TV_P_ROPE] += tv_time_us() - t_rope;
        quantize(row, TV_C, b->s_q, s->q8 + (size_t)t * TV_C);
        quantize(row + TV_C, TV_C, b->s_k, s->k8 + (size_t)t * TV_C);
        quantize(row + 2 * TV_C, TV_C, b->s_v, s->v8 + (size_t)t * TV_C);
        t_rope = tv_time_us();
    }
    snprintf(name, sizeof name, "%s.q", prefix); dbg_heads(c, name, s->q8, T);
    snprintf(name, sizeof name, "%s.k", prefix); dbg_heads(c, name, s->k8, T);
    snprintf(name, sizeof name, "%s.v", prefix); dbg_heads(c, name, s->v8, T);
    for (int h = 0; h < TV_HEADS; h++) {
        int off = h * TV_HEAD_DIM;
        // This head's keys as contiguous rows [T][48], its values transposed [48][Tpad].
        memset(s->vt, 0, (size_t)TV_HEAD_DIM * s->tpad);
        for (int j = 0; j < T; j++) {
            memcpy(s->kh + (size_t)j * TV_HEAD_DIM, s->k8 + (size_t)j * TV_C + off, TV_HEAD_DIM);
            for (int d = 0; d < TV_HEAD_DIM; d++) s->vt[(size_t)d * s->tpad + j] = s->v8[(size_t)j * TV_C + off + d];
        }
        memset(s->w8, 0, s->tpad);
        for (int i = 0; i < T; i++) {
            const int8_t *qi = s->q8 + (size_t)i * TV_C + off;
            int8_t qrow[TV_HEAD_DIM] __attribute__((aligned(16)));
            memcpy(qrow, qi, TV_HEAD_DIM);
            TV_PROF(TV_P_ATTN_DOT, tv_dot_rows_i8(qrow, s->kh, T, TV_HEAD_DIM, s->wts));
            uint64_t t_soft = tv_time_us();
            int32_t mx = INT32_MIN;
            for (int j = 0; j < T; j++) if (s->wts[j] > mx) mx = s->wts[j];
            // Row-max scaled int8 weights (largest = 127), normalized by their own sum.
            int32_t sum = 0, acc[TV_HEAD_DIM];
            const uint8_t *lut = b->softmax_lut;
            const uint32_t lim = (uint32_t)b->softmax_n;
            for (int j = 0; j < T; j++) {
                uint32_t gap = (uint32_t)(mx - s->wts[j]);
                int32_t w = gap < lim ? lut[gap] : 0;
                s->w8[j] = (int8_t)w;
                sum += w;
            }
            tv_prof[TV_P_SOFTMAX] += tv_time_us() - t_soft;
            TV_PROF(TV_P_ATTN_DOT, tv_dot_rows_i8(s->w8, s->vt, TV_HEAD_DIM, s->tpad, acc));
            float k = b->s_v / (float)sum;
            for (int d = 0; d < TV_HEAD_DIM; d++) s->y[(size_t)i * TV_C + off + d] = (float)acc[d] * k;
        }
    }
    quantize(s->y, T * TV_C, b->s_attn, s->i8);
    snprintf(name, sizeof name, "%s.attn", prefix); dbg(c, name, TV_I8, s->i8, T, TV_C);
    linear_f(c, &b->out, s->i8, T, s->acc, out);
}

static void block_forward(ctx *c, const tv_block *b, const char *prefix, float *x, int T, scratch *s) {
    char name[96];
    // x += attention(norm1(x))
    layer_norm(x, T, TV_C, b->norm1_w, b->norm1_b, s->h);
    attention(c, b, prefix, s->h, T, s, s->qkv);        // qkv buffer reused for the output
    for (size_t i = 0; i < (size_t)T * TV_C; i++) x[i] += s->qkv[i];
    // y = depthwise(norm2(x)), float
    layer_norm(x, T, TV_C, b->norm2_w, b->norm2_b, s->h);
    uint64_t t_dw = tv_time_us();
    for (int t = 0; t < T; t++)
        for (int ch = 0; ch < TV_C; ch++) {
            float v = b->dw_b[ch];
            for (int k = 0; k < TV_KERNEL; k++) {
                int src = t + k - TV_KERNEL / 2;
                if (src >= 0 && src < T) v += b->dw_w[ch * TV_KERNEL + k] * s->h[(size_t)src * TV_C + ch];
            }
            s->y[(size_t)t * TV_C + ch] = v;
        }
    tv_prof[TV_P_DW] += tv_time_us() - t_dw;
    quantize(s->y, T * TV_C, b->s_dw, s->i8);
    snprintf(name, sizeof name, "%s.dw", prefix); dbg(c, name, TV_I8, s->i8, T, TV_C);
    // expand -> int8 pre-activation -> GELU table -> project
    linear_f(c, &b->expand, s->i8, T, s->acc, s->h);
    int8_t *pre = s->i8;
    quantize(s->h, T * TV_FF, b->s_pre, pre);
    snprintf(name, sizeof name, "%s.pre", prefix); dbg(c, name, TV_I8, pre, T, TV_FF);
    TV_PROF(TV_P_GELU, for (size_t i = 0; i < (size_t)T * TV_FF; i++) pre[i] = b->gelu_lut[pre[i] + 128]);
    snprintf(name, sizeof name, "%s.gelu", prefix); dbg(c, name, TV_I8, pre, T, TV_FF);
    linear_f(c, &b->project, pre, T, s->acc, s->y);
    for (size_t i = 0; i < (size_t)T * TV_C; i++) x[i] += s->y[i];
}

// x [T][C] is the residual input (overwritten); out receives the final LayerNorm.
static void stack_forward(ctx *c, const tv_stack *st, const char *name, float *x, int T, float *out) {
    scratch s;
    scratch_alloc(&s, T);
    TV_PROF(TV_P_ROPE, rotary(T, s.cos_t, s.sin_t));
    char buf[96];
    for (int i = 0; i < st->count; i++) {
        snprintf(buf, sizeof buf, "%s.blocks.%d", name, i);
        block_forward(c, &st->blocks[i], buf, x, T, &s);
        snprintf(buf, sizeof buf, "%s.%d.out", name, i);
        dbg(c, buf, TV_F32, x, T, TV_C);
    }
    layer_norm(x, T, TV_C, st->norm_w, st->norm_b, out);
    snprintf(buf, sizeof buf, "%s.final", name);
    dbg(c, buf, TV_F32, out, T, TV_C);
    scratch_free(&s);
}

static float gelu_f(float x) { return gelu(x); }

static void predict_durations(ctx *c, const tv_model *m, const float *hidden, const int32_t *ids, int N, float speed,
                              int32_t *dur) {
    float *win = xmalloc((size_t)N * 3 * TV_C * sizeof(float));
    float *h1 = xmalloc((size_t)N * TV_PRED * sizeof(float)), *h2 = xmalloc((size_t)N * TV_PRED * sizeof(float));
    int8_t *q = xmalloc((size_t)N * 3 * TV_C);
    int32_t *acc = xmalloc((size_t)N * TV_PRED * sizeof(int32_t));
    float *out = xmalloc((size_t)N * 2 * sizeof(float));
    windows3(hidden, N, TV_C, win);
    quantize(win, N * 3 * TV_C, m->dur1.in_scale, q);
    dbg(c, "dur.in", TV_I8, q, N, 3 * TV_C);
    linear_f(c, &m->dur1, q, N, acc, h1);
    for (int i = 0; i < N * TV_PRED; i++) h1[i] = gelu_f(h1[i]);
    layer_norm(h1, N, TV_PRED, m->dnorm1_w, m->dnorm1_b, h2);
    windows3(h2, N, TV_PRED, win);
    quantize(win, N * 3 * TV_PRED, m->dur2.in_scale, q);
    dbg(c, "dur.h1", TV_I8, q, N, 3 * TV_PRED);
    linear_f(c, &m->dur2, q, N, acc, h1);
    for (int i = 0; i < N * TV_PRED; i++) h1[i] = gelu_f(h1[i]);
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
    tv_free(win); tv_free(h1); tv_free(h2); tv_free(q); tv_free(acc); tv_free(out);
}

static float gaussian(uint64_t *state) {     // xorshift64* + Box-Muller
    float u[2];
    for (int i = 0; i < 2; i++) {
        *state ^= *state >> 12; *state ^= *state << 25; *state ^= *state >> 27;
        u[i] = ((float)((*state * 0x2545F4914F6CDD1DULL) >> 40) + 0.5f) / 16777216.0f;
    }
    return sqrtf(-2.0f * logf(u[0])) * cosf(6.2831853f * u[1]);
}

long tv_vocode(const tv_model *m, const float *ac, const float *pros, int T, const tv_options *opt, float *audio) {
    ctx c = {opt, 0};
    int8_t *q = xmalloc((size_t)T * TV_C);
    quantize(ac, T * TV_C, m->s_voc, q);
    dbg(&c, "voc.in", TV_I8, q, T, TV_C);
    float *harm = xmalloc((size_t)T * 98 * sizeof(float)), *nz = xmalloc((size_t)T * 36 * sizeof(float));
    float *f0 = xmalloc((size_t)T * sizeof(float));
    int32_t *acc = xmalloc((size_t)T * 98 * sizeof(int32_t));
    linear_f(&c, &m->harmonic, q, T, acc, harm);
    linear_f(&c, &m->noise, q, T, acc, nz);
    for (int f = 0; f < T; f++) {
        float hz = 200.0f * expf(pros[f * 3]);
        f0[f] = hz < 50.0f ? 50.0f : hz > 600.0f ? 600.0f : hz;
    }
    int S = T * TV_HOP;
    float *noise = NULL;
    const float *src = opt->noise;
    if (!src) {
        noise = xmalloc((size_t)S * sizeof(float));
        uint64_t state = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)opt->seed << 1 | 1);
        for (int i = 0; i < S; i++) noise[i] = gaussian(&state);
        src = noise;
    }
    tv_vocoder(harm, nz, f0, src, T, audio);
    dbg(&c, "audio", TV_F32, audio, 1, S);
    tv_free(q); tv_free(harm); tv_free(nz); tv_free(f0); tv_free(acc); tv_free(noise);
    return c.saturations;
}

int tv_synthesize(const tv_model *m, const int32_t *ids, const int32_t *stress, int N, int voice,
                  const tv_options *opt, tv_result *res) {
    ctx c = {opt, 0};
    memset(res, 0, sizeof *res);
    if (N < 1 || voice < 0 || voice > 1) return -1;
    // ---- text encoder
    float *x = xmalloc((size_t)N * TV_C * sizeof(float)), *hidden = xmalloc((size_t)N * TV_C * sizeof(float));
    for (int t = 0; t < N; t++)
        for (int i = 0; i < TV_C; i++)
            x[t * TV_C + i] = m->tokens[ids[t] * TV_C + i] + m->stress[stress[t] * TV_C + i] + m->voices[voice * TV_C + i];
    stack_forward(&c, &m->encoder, "encoder", x, N, hidden);
    // ---- durations
    int32_t *dur = xmalloc((size_t)N * sizeof(int32_t));
    predict_durations(&c, m, hidden, ids, N, opt->speed > 0 ? opt->speed : 1.0f, dur);
    if (opt->force_durations) memcpy(dur, opt->force_durations, N * sizeof(int32_t));
    dbg(&c, "durations", TV_I32, dur, 1, N);
    int T = 0;
    for (int t = 0; t < N; t++) T += dur[t];
    if (T < 1) { tv_free(x); tv_free(hidden); tv_free(dur); return -2; }
    // ---- repeat token states by duration, add position-in-token and voice
    float *frames = xmalloc((size_t)T * TV_C * sizeof(float)), *ph = xmalloc((size_t)T * TV_C * sizeof(float));
    uint8_t *vowel = xmalloc(T);
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
    // ---- prosody
    stack_forward(&c, &m->prosody, "prosody_stack", frames, T, ph);
    float *pros = xmalloc((size_t)T * 3 * sizeof(float));
    float *ac_in = frames;                              // reuse
    for (int f = 0; f < T; f++) {
        const float *row = ph + (size_t)f * TV_C;
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
    dbg(&c, "prosody", TV_F32, pros, T, 3);
    // ---- acoustic
    float *ac = ph;                                     // final LN of the acoustic stack overwrites ph
    float *ac_res = xmalloc((size_t)T * TV_C * sizeof(float));
    memcpy(ac_res, ac_in, (size_t)T * TV_C * sizeof(float));
    stack_forward(&c, &m->acoustic, "acoustic_stack", ac_res, T, ac);
    dbg(&c, "acoustic", TV_F32, ac, T, TV_C);
    // ---- vocoder heads (int8) and source-filter synthesis (float)
    int S = T * TV_HOP;
    res->audio = xmalloc((size_t)S * sizeof(float));
    c.saturations += tv_vocode(m, ac, pros, T, opt, res->audio);
    res->n_samples = S;
    res->n_frames = T;
    res->durations = dur;
    res->qacc_saturations = c.saturations;
    tv_free(x); tv_free(hidden); tv_free(frames); tv_free(ph); tv_free(vowel); tv_free(pros); tv_free(ac_res);
    return 0;
}
