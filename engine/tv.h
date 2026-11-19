// tiny-voca inference engine: portable C, int8 matmuls (W8A8) with the numerics
// specified by tools/tvq.py. The same code runs on the host (reference and
// validation) and on the ESP32-S3 (where the matmul is replaced by PIE kernels).
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TV_C 192            // channels
#define TV_HEADS 4
#define TV_HEAD_DIM 48
#define TV_FF 640
#define TV_KERNEL 7
#define TV_PRED 128         // duration predictor channels
#define TV_HOP 256
#define TV_SAMPLE_RATE 24000
#define TV_QACC_MAX 524287  // QACC lanes are 20-bit signed and saturate
#define TV_VOCAB 76         // token ids 0 .. 75
#define TV_STRESS 3         // stress levels 0 .. 2
#define TV_VOICES 2         // 0 F2, 1 M2
#define TV_ENCODER_BLOCKS 4
#define TV_PROSODY_BLOCKS 2
#define TV_ACOUSTIC_BLOCKS 5
#define TV_BLOCKS (TV_ENCODER_BLOCKS + TV_PROSODY_BLOCKS + TV_ACOUSTIC_BLOCKS)
#define TV_HARM_PARAMS 98   // vocoder harmonic head: 80 causal + 16 anticausal cepstrum, 2 gains
#define TV_NOISE_PARAMS 36  // vocoder noise head: 32 cepstrum, 4 gains

// ---------------------------------------------------------------- container (export.py)
typedef struct {
    char name[64];
    uint32_t dtype, ndim, shape[4], offset, nbytes;
} tv_entry;

typedef struct {
    const uint8_t *base;
    const tv_entry *entries;
    uint32_t count;
} tv_file;

enum { TV_F32 = 0, TV_I8 = 1, TV_I32 = 2 };

int tv_file_open(tv_file *f, const void *data, size_t size);       // 0 on success
const tv_entry *tv_file_find(const tv_file *f, const char *name);
const void *tv_file_get(const tv_file *f, const char *name, uint32_t dtype, uint32_t nelem);  // NULL if missing/mismatch

// ---------------------------------------------------------------- model
typedef struct {
    const int8_t *wq;       // [Npad/16][K][16]
    const float *ws, *b;    // [N]
    const int32_t *pre;     // [chunks][N] centring preload, or NULL
    float *bias;            // [N] bias used by the epilogue (folded constants included)
    int N, K, chunks;
    float in_scale;         // scale of the int8 input
    void *backend;          // kernel-backend data (e.g. packed QACC preloads)
} tv_linear;

typedef struct {
    const float *norm1_w, *norm1_b, *norm2_w, *norm2_b, *dw_w, *dw_b;
    tv_linear qkv, out, expand, project;
    float s_ln1, s_q, s_k, s_v, s_attn, s_dw, s_pre, s_gelu;  // int8 scales (gelu: asymmetric step)
    int8_t gelu_lut[256];   // pre code (-127..127) + 128 -> gelu code (-128..127)
    uint8_t *softmax_lut;   // [gap] = round(127 * exp(-gap * score_scale)), gap = max dot - dot
    int softmax_n;          // table length; larger gaps give weight 0
} tv_block;

typedef struct {
    tv_block *blocks;
    int count;
    const float *norm_w, *norm_b;
} tv_stack;

typedef struct {
    tv_file file;
    const float *tokens, *stress, *voices, *frame_voices;   // embeddings
    tv_stack encoder, prosody, acoustic;
    tv_block block_storage[TV_BLOCKS];
    // duration predictor
    tv_linear dur1, dur2;
    const float *dnorm1_w, *dnorm1_b, *dnorm2_w, *dnorm2_b, *dur_out_w, *dur_out_b;
    // frame-level
    const float *frame_pos_w, *frame_pos_b, *prosody_head_w, *prosody_head_b, *prosody_proj_w, *prosody_proj_b;
    // vocoder heads
    tv_linear harmonic, noise;
    float s_voc;
    char error[128];
} tv_model;

int tv_model_init(tv_model *m, const void *data, size_t size);   // 0 on success, else see m->error
void tv_model_free(tv_model *m);

// ---------------------------------------------------------------- inference
// Audio sink for streaming output: called with consecutive chunks of samples.
typedef void (*tv_audio_fn)(void *user, const float *samples, int n);

// Debug hook: called with intermediate tensors (names as in the golden dump).
typedef void (*tv_debug_fn)(void *user, const char *name, int dtype, const void *data, int rows, int cols);

typedef struct {
    float speed;                    // 1.0
    const float *noise;             // NULL: engine's own Gaussian noise; else noise_len values,
    int noise_len;                  // at least n_frames * TV_HOP (else TV_BAD_INPUT)
    uint32_t seed;
    const int32_t *force_durations; // NULL: predicted
    int max_frames;                 // > 0: return TV_TOO_LONG instead of synthesizing more frames
    // The acoustic stack is streamed into the vocoder. With a look-ahead > 0 a frame sees only that
    // many later frames, so sound starts before the whole stack is done; with past > 0 (only used with
    // a look-ahead) also only that many earlier ones, so its memory does not grow with the utterance.
    // A look-ahead without a past keeps whole-utterance keys and values for every block. 0: full
    // attention. Debug runs compute the same in one pass, with every tensor whole.
    int acoustic_lookahead;
    int acoustic_past;
    tv_audio_fn on_audio;           // set: audio is streamed here and tv_result.audio stays NULL
    void *audio_user;
    tv_debug_fn debug;
    void *debug_user;
} tv_options;

typedef struct {
    float *audio;                   // n_samples (caller frees with tv_free)
    int n_samples, n_frames;
    int32_t *durations;             // n_tokens (caller frees with tv_free)
    long qacc_saturations;          // must be 0
} tv_result;

// tv_synthesize returns 0 or one of these (tv_result is then empty, but for n_frames of TV_TOO_LONG).
#define TV_BAD_INPUT (-1)           // no tokens, an id, stress, voice or forced duration out of range,
                                    // or too little noise
#define TV_NO_FRAMES (-2)           // every duration is 0
#define TV_TOO_LONG (-3)            // predicted frames exceed tv_options.max_frames; split the text
int tv_synthesize(const tv_model *m, const int32_t *ids, const int32_t *stress, int n_tokens, int voice,
                  const tv_options *opt, tv_result *out);

// Last stage alone: acoustic features [T][192] + prosody [T][3] -> audio [T*hop] (or opt->on_audio).
// Returns saturations.
long tv_vocode(const tv_model *m, const float *acoustic, const float *prosody, int T, const tv_options *opt, float *audio);

// ---------------------------------------------------------------- profiling
void tv_profile_reset(void);
void tv_profile_report(void);       // printf, one line per op category (time spent since the reset)

#ifdef __cplusplus
}
#endif
