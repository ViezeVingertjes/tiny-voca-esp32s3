// Source-filter vocoder of tiny-voca (model.py SourceFilter), batch size 1.
// Matches torch.stft/istft with center=True, constant padding, periodic Hann window.
#include "tv.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define SYNTH_FFT 512
#define SYNTH_HOP 128
#define NOISE_FFT 256
#define NOISE_HOP 64
#define CEP_CAUSAL 80
#define CEP_ANTI 16
#define NOISE_CEP 32
#define HARMONIC_LIMIT 0.47

// ---------------------------------------------------------------- fast float math
// exp: Cody-Waite reduction + degree-6 polynomial (~2 ulp). sincos: quadrant reduction + minimax.
static inline float fast_expf(float x) {
    if (x < -87.0f) x = -87.0f;
    if (x > 88.0f) x = 88.0f;
    float n = rintf(x * 1.44269504f);
    float r = x - n * 0.693145752f - n * 1.42860677e-6f;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.166666672f + r * (0.0416666679f + r * (0.00833333377f + r * 0.00138888892f)))));
    union { float f; int32_t i; } u = {p};
    u.i += (int32_t)n << 23;
    return u.f;
}

static inline void fast_sincosf(float x, float *s, float *c) {
    float k = rintf(x * 0.636619772f);                  // quadrant
    float r = x - k * 1.57079601f - k * 3.13916473e-07f - k * 5.39030253e-15f;
    float r2 = r * r;
    float sn = r + r * r2 * (-0.166666657f + r2 * (0.00833330024f + r2 * -0.000198409713f));
    float cs = 1.0f + r2 * (-0.5f + r2 * (0.0416666418f + r2 * (-0.00138867637f + r2 * 2.43904487e-05f)));
    int q = (int)k & 3;
    *s = q == 0 ? sn : q == 1 ? cs : q == 2 ? -sn : -cs;
    *c = q == 0 ? cs : q == 1 ? -sn : q == 2 ? -cs : sn;
}

// ---------------------------------------------------------------- FFT with cached tables
// Complex radix-2 FFT of size m (128 or 256) used for real FFTs of size 2m.
typedef struct {
    int m;
    float *tw_re, *tw_im;       // exp(-2 pi i j / m), j < m/2
    float *rw_re, *rw_im;       // exp(-2 pi i k / 2m), k <= m (real-FFT split)
    uint16_t *rev;
} fft_plan;

static fft_plan plans[2];

static const fft_plan *plan_for(int n) {           // n = real size (256 or 512)
    int m = n / 2;
    fft_plan *p = &plans[m == 256];
    if (p->m == m) return p;
    p->tw_re = malloc(m / 2 * sizeof(float)); p->tw_im = malloc(m / 2 * sizeof(float));
    p->rw_re = malloc((m + 1) * sizeof(float)); p->rw_im = malloc((m + 1) * sizeof(float));
    p->rev = malloc(m * sizeof(uint16_t));
    for (int j = 0; j < m / 2; j++) { p->tw_re[j] = (float)cos(-2 * M_PI * j / m); p->tw_im[j] = (float)sin(-2 * M_PI * j / m); }
    for (int k = 0; k <= m; k++) { p->rw_re[k] = (float)cos(-2 * M_PI * k / n); p->rw_im[k] = (float)sin(-2 * M_PI * k / n); }
    int bits = 0;
    while ((1 << bits) < m) bits++;
    for (int i = 0; i < m; i++) {
        int r = 0;
        for (int b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
        p->rev[i] = (uint16_t)r;
    }
    p->m = m;
    return p;
}

// In place, unscaled; inverse uses conjugate twiddles.
static void cfft(const fft_plan *p, float *re, float *im, int inverse) {
    int m = p->m;
    for (int i = 0; i < m; i++) {
        int j = p->rev[i];
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    const float sign = inverse ? -1.0f : 1.0f;
    for (int len = 2, step = m / 2; len <= m; len <<= 1, step >>= 1) {
        int half = len / 2;
        for (int i = 0; i < m; i += len)
            for (int k = 0; k < half; k++) {
                float wr = p->tw_re[k * step], wi = sign * p->tw_im[k * step];
                int a = i + k, b = a + half;
                float tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr; im[a] += ti;
            }
    }
}

// rfft of n real values -> n/2+1 bins, via a complex FFT of size n/2.
static void rfft(const float *x, int n, float *re, float *im, float *zr, float *zi) {
    const fft_plan *p = plan_for(n);
    int m = n / 2;
    for (int k = 0; k < m; k++) { zr[k] = x[2 * k]; zi[k] = x[2 * k + 1]; }
    cfft(p, zr, zi, 0);
    for (int k = 0; k <= m; k++) {
        int a = k % m, b = (m - k) % m;
        float er = 0.5f * (zr[a] + zr[b]), ei = 0.5f * (zi[a] - zi[b]);      // (Z[k] + conj Z[m-k]) / 2
        float orr = 0.5f * (zi[a] + zi[b]), oi = -0.5f * (zr[a] - zr[b]);    // (Z[k] - conj Z[m-k]) / 2i
        float wr = p->rw_re[k], wi = p->rw_im[k];
        re[k] = er + wr * orr - wi * oi;
        im[k] = ei + wr * oi + wi * orr;
    }
}

// irfft: n/2+1 bins -> n real values (imaginary parts of DC and Nyquist ignored, like torch).
static void irfft(const float *re, const float *im, int n, float *x, float *zr, float *zi) {
    const fft_plan *p = plan_for(n);
    int m = n / 2;
    for (int k = 0; k < m; k++) {
        float ar = re[k], ai = k ? im[k] : 0.0f;
        float br = re[m - k], bi = (m - k) == m ? 0.0f : -im[m - k];         // conj X[m-k]
        if (k == 0) bi = 0.0f;
        float er = 0.5f * (ar + br), ei = 0.5f * (ai + bi);
        float dr = 0.5f * (ar - br), di = 0.5f * (ai - bi);
        float wr = p->rw_re[k], wi = -p->rw_im[k];                            // conj W^k
        float orr = dr * wr - di * wi, oi = dr * wi + di * wr;
        zr[k] = er - oi;                                                      // E + i O
        zi[k] = ei + orr;
    }
    cfft(p, zr, zi, 1);
    float inv = 1.0f / m;
    for (int k = 0; k < m; k++) { x[2 * k] = zr[k] * inv; x[2 * k + 1] = zi[k] * inv; }
}

// exp(FFT(cepstrum)) as a complex response: causal part at the start of the buffer,
// anticausal part (reversed) at its end.
static void cepstrum_response(const float *causal, int nc, const float *anti, int na, int n,
                              float *re, float *im, float *buf, float *tr, float *ti) {
    memset(buf, 0, n * sizeof(float));
    memcpy(buf, causal, nc * sizeof(float));
    for (int i = 0; i < na; i++) buf[n - na + i] = anti[na - 1 - i];
    rfft(buf, n, re, im, tr, ti);
    for (int k = 0; k <= n / 2; k++) {
        float mag = fast_expf(re[k] < -14 ? -14 : re[k] > 6 ? 6 : re[k]);
        float sn, cs;
        fast_sincosf(im[k], &sn, &cs);
        re[k] = mag * cs;
        im[k] = mag * sn;
    }
}

static float lifter(int i, int first) {
    if (first == 0 && i == 0) return 2.0f;
    return 1.0f / (1.0f + (float)(first + i) / 4.0f);
}

// Time-varying filter: STFT of `src` (length S, followed by fft/2 zeros), multiply frame m by
// response m, inverse STFT, keep the first S samples. resp_re/im: [frames][fft/2+1].
static void shape(const float *src, int S, int fft_n, int hop, const float *resp_re, const float *resp_im, float *out) {
    int L = S + fft_n / 2;                      // padded source length
    int frames = 1 + L / hop;
    int P = fft_n / 2, total = L + 2 * P;       // center padding on both sides
    int bins = fft_n / 2 + 1;
    float *padded = calloc(total, sizeof(float)), *ola = calloc(total, sizeof(float)), *env = calloc(total, sizeof(float));
    float *win = malloc(fft_n * sizeof(float)), *seg = malloc(fft_n * sizeof(float));
    float *re = malloc(bins * sizeof(float)), *im = malloc(bins * sizeof(float));
    float *tr = malloc(fft_n * sizeof(float)), *ti = malloc(fft_n * sizeof(float));
    memcpy(padded + P, src, S * sizeof(float));
    for (int i = 0; i < fft_n; i++) win[i] = (float)(0.5 - 0.5 * cos(2 * M_PI * i / fft_n));
    for (int m = 0; m < frames; m++) {
        const float *x = padded + m * hop;
        for (int i = 0; i < fft_n; i++) seg[i] = x[i] * win[i];
        rfft(seg, fft_n, re, im, tr, ti);
        const float *rr = resp_re + (size_t)m * bins, *ri = resp_im + (size_t)m * bins;
        for (int k = 0; k < bins; k++) {
            float a = re[k] * rr[k] - im[k] * ri[k], b = re[k] * ri[k] + im[k] * rr[k];
            re[k] = a; im[k] = b;
        }
        irfft(re, im, fft_n, seg, tr, ti);
        for (int i = 0; i < fft_n; i++) {
            ola[m * hop + i] += seg[i] * win[i];
            env[m * hop + i] += win[i] * win[i];
        }
    }
    for (int i = 0; i < S; i++) {
        float e = env[P + i];
        out[i] = e > 1e-11f ? ola[P + i] / e : ola[P + i];
    }
    free(padded); free(ola); free(env); free(win); free(seg); free(re); free(im); free(tr); free(ti);
}

// Band-limited pulse train with unit power (model.py pulse_train). The phase is kept in
// double cycles (exact enough over minutes); everything else is float.
static void pulse_train(const float *f0, int T, float *out) {
    double cycles = 0;
    for (int f = 0; f < T; f++) {
        float k0 = logf(f0[f] < 1 ? 1.0f : f0[f]);
        float k1 = logf(f0[f + 1 < T ? f + 1 : T - 1] < 1 ? 1.0f : f0[f + 1 < T ? f + 1 : T - 1]);
        for (int i = 0; i < TV_HOP; i++) {
            float w = (float)i / TV_HOP;
            float freq = fast_expf(k0 + (k1 - k0) * w);
            cycles += freq * (1.0f / TV_SAMPLE_RATE);
            if (cycles >= 1.0) cycles -= 1.0;
            float phase = (float)cycles;                     // in [0, 1)
            float th = 6.28318531f * phase;
            float harmonics = (float)HARMONIC_LIMIT * TV_SAMPLE_RATE / freq;
            if (harmonics < 1) harmonics = 1;
            float whole = floorf(harmonics);
            float s_half, c_half, s_core, c_core, s_top, c_top;
            fast_sincosf(0.5f * th, &s_half, &c_half);
            float u = (whole + 0.5f) * phase, v = (whole + 1.0f) * phase;     // reduce in cycles first
            fast_sincosf(6.28318531f * (u - floorf(u)), &s_core, &c_core);
            fast_sincosf(6.28318531f * (v - floorf(v)), &s_top, &c_top);
            float den = 2.0f * s_half;
            float core = fabsf(den) < 1e-7f ? whole : s_core / den - 0.5f;
            float wave = core + (harmonics - whole) * c_top;
            out[f * TV_HOP + i] = wave * sqrtf(2.0f / harmonics);
        }
    }
}

void tv_vocoder(const float *h, const float *n, const float *f0, const float *noise, int T, float *audio) {
    const int S = T * TV_HOP, hsteps = TV_HOP / SYNTH_HOP, nsteps = TV_HOP / NOISE_HOP;
    const int hb = SYNTH_FFT / 2 + 1, nb = NOISE_FFT / 2 + 1;
    const int hframes = hsteps * T + 3, nframes = nsteps * T + 3;
    float *hre = malloc((size_t)hframes * hb * sizeof(float)), *him = malloc((size_t)hframes * hb * sizeof(float));
    float *nre = malloc((size_t)nframes * nb * sizeof(float)), *nim = malloc((size_t)nframes * nb * sizeof(float));
    float *buf = malloc(SYNTH_FFT * sizeof(float)), *tr = malloc(SYNTH_FFT * sizeof(float)), *ti = malloc(SYNTH_FFT * sizeof(float));
    float causal[CEP_CAUSAL], anti[CEP_ANTI], ncep[NOISE_CEP];
    uint64_t t_cep = tv_time_us();
    // Harmonic branch: envelope interpolated between frames, one gain per sub-frame.
    for (int j = 0; j < hframes; j++) {
        int sub = j < hsteps * T - 1 ? j : hsteps * T - 1;
        int here = sub / hsteps, next = (sub + 1) / hsteps < T - 1 ? (sub + 1) / hsteps : T - 1;
        const float *a = h + (size_t)here * 98, *b = h + (size_t)next * 98;
        for (int i = 0; i < CEP_CAUSAL; i++) causal[i] = 0.5f * (a[i] + b[i]) * lifter(i, 0);
        for (int i = 0; i < CEP_ANTI; i++) anti[i] = 0.5f * (a[CEP_CAUSAL + i] + b[CEP_CAUSAL + i]) * lifter(i, 1);
        causal[0] += h[(size_t)(sub / hsteps) * 98 + CEP_CAUSAL + CEP_ANTI + sub % hsteps];
        cepstrum_response(causal, CEP_CAUSAL, anti, CEP_ANTI, SYNTH_FFT, hre + (size_t)j * hb, him + (size_t)j * hb, buf, tr, ti);
    }
    // Noise branch: one envelope per frame, four gains per frame.
    for (int j = 0; j < nframes; j++) {
        int sub = j < nsteps * T - 1 ? j : nsteps * T - 1;
        const float *a = n + (size_t)(sub / nsteps) * 36;
        for (int i = 0; i < NOISE_CEP; i++) ncep[i] = a[i] * lifter(i, 0);
        ncep[0] += a[NOISE_CEP + sub % nsteps];
        cepstrum_response(ncep, NOISE_CEP, NULL, 0, NOISE_FFT, nre + (size_t)j * nb, nim + (size_t)j * nb, buf, tr, ti);
    }
    tv_prof[TV_P_VOC_CEP] += tv_time_us() - t_cep;
    float *pulses = malloc((size_t)S * sizeof(float)), *part = malloc((size_t)S * sizeof(float));
    TV_PROF(TV_P_VOC_PULSE, pulse_train(f0, T, pulses));
    TV_PROF(TV_P_VOC_SHAPE, shape(pulses, S, SYNTH_FFT, SYNTH_HOP, hre, him, audio));
    TV_PROF(TV_P_VOC_SHAPE, shape(noise, S, NOISE_FFT, NOISE_HOP, nre, nim, part));
    for (int i = 0; i < S; i++) audio[i] += part[i];
    free(hre); free(him); free(nre); free(nim); free(buf); free(tr); free(ti); free(pulses); free(part);
}
