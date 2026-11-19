// tiny-voca on the ESP32-S3: a console that speaks typed text (phonemizer, synthesis, speaker),
// with the model mapped from the `model` partition. Commands for the tools: TEST runs the test
// vector from the `testvec` partition and streams prosody + audio back as base64 ("TVDUMP"
// blocks, decoded by tools/serial_dump.py); SYN synthesizes given tokens (tools/speak_s3.py).
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tv.h"
#include "tv_kernels.h"
#include "board.h"
#include "phonemize.h"
#include "sentences.h"

static const void *map_partition(const char *label, size_t *size) {
    const esp_partition_t *p = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, label);
    if (!p) { printf("partition %s not found\n", label); return NULL; }
    const void *ptr;
    esp_partition_mmap_handle_t handle;
    if (esp_partition_mmap(p, 0, p->size, ESP_PARTITION_MMAP_DATA, &ptr, &handle) != ESP_OK) {
        printf("mmap of %s failed\n", label);
        return NULL;
    }
    *size = p->size;
    return ptr;
}

static void dump_base64(const char *name, const void *data, size_t n, int rows, int cols) {
    static const char tab[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const uint8_t *p = data;
    printf("TVDUMP %s %d %d\n", name, rows, cols);
    char line[97];
    int k = 0;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = p[i] << 16 | (i + 1 < n ? p[i + 1] << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
        line[k++] = tab[v >> 18 & 63];
        line[k++] = tab[v >> 12 & 63];
        line[k++] = i + 1 < n ? tab[v >> 6 & 63] : '=';
        line[k++] = i + 2 < n ? tab[v & 63] : '=';
        if (k == 96 || i + 3 >= n) { line[k] = 0; printf("%s\n", line); k = 0; }
    }
    printf("TVDUMP_END %s\n", name);
}

typedef struct {
    int64_t t0;
    const int32_t *golden_durations;
    float *prosody;
    int prosody_rows;
} run_state;

static void on_tensor(void *user, const char *name, int dtype, const void *data, int rows, int cols) {
    run_state *s = user;
    static const char *stages[] = {"encoder.final", "durations", "prosody_stack.final", "acoustic_stack.final", "audio"};
    for (int i = 0; i < 5; i++)
        if (!strcmp(name, stages[i])) printf("  %-22s at %7.1f ms\n", name, (esp_timer_get_time() - s->t0) / 1000.0);
    if (!strcmp(name, "durations")) {
        int differ = 0;
        for (int i = 0; i < cols; i++) differ += ((const int32_t *)data)[i] != s->golden_durations[i];
        printf("  durations vs golden: %d of %d differ\n", differ, cols);
    }
    if (!strcmp(name, "prosody")) {
        s->prosody = tv_alloc((size_t)rows * cols * 4);
        memcpy(s->prosody, data, (size_t)rows * cols * 4);
        s->prosody_rows = rows;
    }
}

#define MAX_TOKENS 512             // phonemes per utterance (the console splits longer text)

static tv_model model;
static tv_file test;
static int have_test;

static void run_test(void) {
    if (!have_test) { printf("no test vector\n"); return; }
    const tv_entry *ids_e = tv_file_find(&test, "ids"), *noise_e = tv_file_find(&test, "noise");
    int n = ids_e ? (int)ids_e->shape[0] : 0;
    const int32_t *ids = tv_file_get(&test, "ids", TV_I32, n), *stress = tv_file_get(&test, "stress", TV_I32, n);
    const int32_t *voice = tv_file_get(&test, "voice", TV_I32, 1);
    run_state st = {.golden_durations = tv_file_get(&test, "durations", TV_I32, n)};
    if (!ids || !stress || !voice || !noise_e || !st.golden_durations) { printf("no test vector (testvec is not test0_dev.bin)\n"); return; }
    tv_options opt = {.speed = 1.0f, .noise = (const float *)(test.base + noise_e->offset), .noise_len = noise_e->nbytes / 4,
                      .debug = on_tensor, .debug_user = &st};
    tv_result r;
    tv_profile_reset();
    st.t0 = esp_timer_get_time();
    int rc = tv_synthesize(&model, ids, stress, n, *voice, &opt, &r);
    double ms = (esp_timer_get_time() - st.t0) / 1000.0;
    if (rc) { printf("synthesize failed: %d\n", rc); return; }
    double audio_ms = 1000.0 * r.n_samples / TV_SAMPLE_RATE;
    tv_profile_report();
    printf("RESULT %d frames, %.0f ms of audio in %.0f ms (%.2fx real time), QACC saturations %ld\n", r.n_frames,
           audio_ms, ms, audio_ms / ms, r.qacc_saturations);
    {   // the production path (no debug hook: fused epilogues) must give the same audio
        tv_options o2 = opt;
        o2.debug = NULL;
        tv_result r2;
        int64_t t1 = esp_timer_get_time();
        if (tv_synthesize(&model, ids, stress, n, *voice, &o2, &r2) == 0) {
            int same = r2.n_samples == r.n_samples && !memcmp(r2.audio, r.audio, (size_t)r.n_samples * 4);
            printf("production path: %.0f ms, audio %s\n", (esp_timer_get_time() - t1) / 1000.0, same ? "bit-identical" : "DIFFERS");
            tv_free(r2.audio); tv_free(r2.durations);
        }
    }
    if (st.prosody) dump_base64("prosody", st.prosody, (size_t)st.prosody_rows * 12, st.prosody_rows, 3);
    dump_base64("audio", r.audio, (size_t)r.n_samples * 4, 1, r.n_samples);
    tv_free(st.prosody);
    tv_free(r.audio); tv_free(r.durations);
    printf("TVDUMP_DONE\n");
}

// The acoustic stack is streamed into the vocoder: a frame sees 32 frames ahead and 512 back, so
// sound starts early and its memory stays fixed. "look 0" switches to full attention (the
// reference numerics; sound starts only once the whole acoustic stack is done).
static int acoustic_lookahead = 32;
#define ACOUSTIC_PAST 512

// Frames that fit in the free PSRAM. Streaming with a past window: ~2.7 MB fixed (key and value
// rings, the token states; 3 MB with a margin) plus ~1.3 KB per frame, measured; full attention:
// ~2.9 KB per frame. The largest single buffer is T x 768 bytes. keep_audio: the whole utterance
// is returned as float audio (1 KB per frame, then one buffer) and converted to 16 bits (0.5 KB).
// The model itself handles at most 3000 frames.
static int frame_budget(int lookahead, int keep_audio) {
    long fixed = lookahead > 0 ? 3000 * 1024 : 1200 * 1024, bytes_per_frame = lookahead > 0 ? 1300 : 2900;
    long largest_per_frame = 800;
    if (keep_audio) { bytes_per_frame += 1536; largest_per_frame = 1100; }
    long total = (long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM) - fixed;
    long largest = (long)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) - 64 * 1024;
    long frames = total / bytes_per_frame;
    if (largest / largest_per_frame < frames) frames = largest / largest_per_frame;
    if (frames > 3000) frames = 3000;
    return frames > 0 ? (int)frames : 0;
}

// SYN <voice> <seed> <n> <id,id,...> <stress,stress,...>
static void run_syn(char *line) {
    int voice, n;
    unsigned seed;
    char *rest;
    if (sscanf(line, "SYN %d %u %d", &voice, &seed, &n) != 3 || n < 1 || n > MAX_TOKENS) { printf("ERR bad command\n"); return; }
    rest = line;
    for (int i = 0; i < 4 && rest; i++) rest = strchr(rest + 1, ' ');
    int32_t *ids = tv_alloc(n * 4), *stress = tv_alloc(n * 4);
    char *tok = rest ? strtok(rest + 1, " ") : NULL;
    char *ids_s = tok, *str_s = tok ? strtok(NULL, " \r\n") : NULL;
    int ok = ids_s && str_s;
    for (int i = 0; ok && i < n; i++) {     // exactly n comma-separated values in each list
        char *a = ids_s, *b = str_s;
        ids[i] = strtol(a, &ids_s, 10);
        stress[i] = strtol(b, &str_s, 10);
        ok = ids_s != a && str_s != b && ids[i] >= 0 && ids[i] < TV_VOCAB && stress[i] >= 0 && stress[i] < TV_STRESS;
        if (ok && i < n - 1) { ok = *ids_s == ',' && *str_s == ','; ids_s++; str_s++; }
    }
    if (ok) ok = !*ids_s && !*str_s;
    if (!ok) { printf("ERR bad tokens\n"); tv_free(ids); tv_free(stress); return; }
    tv_options opt = {.speed = 1.0f, .seed = seed, .max_frames = frame_budget(0, 1)};
    tv_result r;
    tv_profile_reset();
    int64_t t0 = esp_timer_get_time();
    int rc = tv_synthesize(&model, ids, stress, n, voice, &opt, &r);
    double ms = (esp_timer_get_time() - t0) / 1000.0;
    tv_free(ids); tv_free(stress);
    if (rc) { printf("ERR synthesize %d\n", rc); return; }
    printf("RESULT tokens=%d frames=%d audio_ms=%.1f synth_ms=%.1f saturations=%ld free_psram=%u\n", n, r.n_frames,
           1000.0 * r.n_samples / TV_SAMPLE_RATE, ms, r.qacc_saturations, (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    tv_profile_report();
    int16_t *pcm = tv_alloc((size_t)r.n_samples * 2);
    for (int i = 0; i < r.n_samples; i++) {
        float v = r.audio[i] * 32767.0f;
        pcm[i] = (int16_t)(v > 32767 ? 32767 : v < -32767 ? -32767 : v);
    }
    dump_base64("pcm", pcm, (size_t)r.n_samples * 2, 1, r.n_samples);
    tv_free(pcm); tv_free(r.audio); tv_free(r.durations);
    printf("DONE\n");
}

// Streams synthesized audio to the speaker in 0.2 s chunks (clipped at +-0.99).
typedef struct {
    int16_t *chunk;
    int fill;
    float peak;
    int64_t t_first;            // when playback started
    int preroll;                // samples to collect before playback starts
    int64_t play_end;           // when the queued audio runs out
    int gaps;                   // chunks that came after it ran out (audible gaps)
    int64_t gap_us;
} stream_state;

#define STREAM_CHUNK 4800

static void stream_flush(stream_state *st) {
    if (!st->fill) return;
    int64_t now = esp_timer_get_time();
    if (!st->t_first) st->t_first = now;
    if (st->play_end && now > st->play_end) { st->gaps++; st->gap_us += now - st->play_end; }
    st->play_end = (now > st->play_end ? now : st->play_end) + (int64_t)st->fill * 1000000 / TV_SAMPLE_RATE;
    board_play(st->chunk, st->fill);                 // the player frees it
    st->chunk = NULL;
    st->fill = 0;
}

static void on_audio(void *user, const float *x, int n) {
    stream_state *st = user;
    for (int i = 0; i < n; i++) {
        int size = st->t_first ? STREAM_CHUNK : st->preroll;
        if (!st->chunk) st->chunk = tv_alloc((size_t)size * 2);
        float v = x[i];
        if (fabsf(v) > st->peak) st->peak = fabsf(v);
        v = v > 0.99f ? 0.99f : v < -0.99f ? -0.99f : v;
        st->chunk[st->fill++] = (int16_t)lrintf(v * 32767.0f);
        if (st->fill == size) stream_flush(st);
    }
}

static int profiling = 0;

// Matmul placement experiment on a real layer (acoustic block 0 expand: 640 x 192).
static void mmbench(void) {
    tv_linear L = model.block_storage[6].expand;
    const int K = L.K, N = L.N, npad = (N + 15) / 16 * 16;
    int8_t *w_ps = heap_caps_aligned_alloc(16, (size_t)npad * K + 64, MALLOC_CAP_SPIRAM);
    memcpy(w_ps, L.wq, (size_t)npad * K);
    for (int rows = 16; rows <= 64; rows *= 4) {
        int8_t *x_ps = tv_alloc((size_t)rows * K), *x_in = heap_caps_aligned_alloc(16, (size_t)rows * K + 64, MALLOC_CAP_INTERNAL);
        float *y_ps = tv_alloc((size_t)rows * N * 4), *y_in = rows <= 16 ? heap_caps_malloc((size_t)rows * N * 4, MALLOC_CAP_INTERNAL) : NULL;
        for (int i = 0; i < rows * K; i++) x_ps[i] = x_in[i] = (int8_t)((i * 37) % 127 - 63);
        struct { const char *name; const int8_t *w, *x; float *y; } cfg[] = {
            {"W flash, x PSRAM, y PSRAM", L.wq, x_ps, y_ps}, {"W PSRAM, x PSRAM, y PSRAM", w_ps, x_ps, y_ps},
            {"W flash, x SRAM,  y PSRAM", L.wq, x_in, y_ps}, {"W flash, x SRAM,  y SRAM", L.wq, x_in, y_in},
            {"W PSRAM, x SRAM,  y SRAM", w_ps, x_in, y_in}};
        for (int c = 0; c < 5; c++) {
            if (!cfg[c].y) continue;
            tv_linear l2 = L;
            l2.wq = cfg[c].w;
            int64_t best = INT64_MAX;
            for (int rep = 0; rep < 3; rep++) {
                int64_t t0 = esp_timer_get_time();
                tv_epilogue e = {TV_EPI_STORE, 0, NULL};
                tv_linear_run(&l2, cfg[c].x, rows, cfg[c].y, &e);
                int64_t dt = esp_timer_get_time() - t0;
                if (dt < best) best = dt;
            }
            printf("  rows %2d  %-28s %6lld us  %6.0f MMAC/s\n", rows, cfg[c].name, (long long)best, (double)rows * N * K / best);
        }
        tv_free(x_ps); heap_caps_free(x_in); tv_free(y_ps); if (y_in) heap_caps_free(y_in);
    }
    heap_caps_free(w_ps);
}

static int32_t tok_ids[MAX_TOKENS], tok_stress[MAX_TOKENS];     // phonemizer output (say_piece, PHON)

// One piece of text: phonemize, synthesize, queue for playback. Splits at clauses when too long.
static int say_piece(char *text, int voice, unsigned seed, int depth) {
    int32_t *ids = tok_ids, *stress = tok_stress;
    int n = tvp_phonemize(text, ids, stress, MAX_TOKENS);
    if (n == -2 || n > 0) {
        // Streaming a longer utterance, the acoustic stack still has work to do while the first audio
        // plays: collect 0.4 s before playback starts so the next chunk is ready in time. Up to
        // ~250 frames (~30 phonemes) that was never needed.
        stream_state st = {.preroll = acoustic_lookahead > 0 && n > 30 ? 2 * STREAM_CHUNK : STREAM_CHUNK};
        board_show_text(text);
        tv_profile_reset();
        tv_options opt = {.speed = 1.0f, .seed = seed, .max_frames = frame_budget(acoustic_lookahead, 0), .on_audio = on_audio,
                          .audio_user = &st, .acoustic_lookahead = acoustic_lookahead, .acoustic_past = ACOUSTIC_PAST};
        tv_result r = {0};
        int64_t t0 = esp_timer_get_time();
        int rc = n > 0 ? tv_synthesize(&model, ids, stress, n, voice, &opt, &r) : TV_TOO_LONG;
        if (rc == 0) {
            stream_flush(&st);
            double ms = (esp_timer_get_time() - t0) / 1000.0, audio_ms = 1000.0 * r.n_samples / TV_SAMPLE_RATE;
            char gaps[48] = "";
            if (st.gaps) snprintf(gaps, sizeof gaps, ", %d gaps (%.2f s)", st.gaps, st.gap_us / 1e6);
            printf("  [%d phonemes, %d frames, %.2f s audio, first sound after %.2f s, done after %.2f s, peak %.2f%s] %s\n",
                   n, r.n_frames, audio_ms / 1000, (st.t_first - t0) / 1e6, ms / 1000, st.peak, gaps, text);
            tv_free(r.durations);
            if (profiling) tv_profile_report();
            return 0;
        }
        if (rc != TV_TOO_LONG) { printf("  synthesis failed (%d): %s\n", rc, text); return -1; }
        if (depth == 0) {
            char *parts[32];
            int k = tv_split_clauses(text, parts, 32);
            if (k > 1) {
                for (int i = 0; i < k; i++) say_piece(parts[i], voice, seed + 101 * i, 1);
                return 0;
            }
        }
        if (n < 0) printf("  cannot phonemize: %s (%s)\n", tvp_error(), text);
        else printf("  too long for memory (%d frames, budget %d): %s\n", r.n_frames, opt.max_frames, text);
        return -1;
    }
    printf("  cannot phonemize: %s (%s)\n", tvp_error(), text);
    return -1;
}

static int voice_id = 1;      // M2

// line (split in place) must have room for one more byte (tv_finish_text).
static void say(char *line) {
    tv_finish_text(line);
    char *parts[64];
    int k = tv_split_sentences(line, parts, 64);
    for (int i = 0; i < k; i++) say_piece(parts[i], voice_id, (unsigned)i, 0);
}

// Line input with echo and backspace (works in idf.py monitor and from scripts).
// An empty line returns 0: the tools send one to wait for READY.
static int read_line(char *buf, int size) {
    static int prev;
    int n = 0;
    for (;;) {
        int c = fgetc(stdin);
        if (c == EOF) { clearerr(stdin); vTaskDelay(pdMS_TO_TICKS(10)); continue; }
        int last = prev;
        prev = c;
        if (c == '\n' && last == '\r') continue;    // second half of a CRLF
        if (c == '\r' || c == '\n') {
            putchar('\n'); fflush(stdout);
            buf[n] = 0;
            return n;
        }
        if (c == 0x7f || c == 0x08) {
            if (n) { n--; fputs("\b \b", stdout); fflush(stdout); }
            continue;
        }
        if (c == '\t') c = ' ';
        if (n < size - 1 && c >= 0x20) {
            buf[n++] = (char)c;
            putchar(c); fflush(stdout);
        }
    }
}

static void run(void *arg) {
    size_t msize, tsize, psize;
    const void *mdata = map_partition("model", &msize), *tdata = map_partition("testvec", &tsize);
    const void *pdata = map_partition("phonemizer", &psize);
    if (!mdata) goto done;
    int64_t t = esp_timer_get_time();
    if (tv_model_init(&model, mdata, msize)) { printf("model init failed: %s\n", model.error); goto done; }
    printf("model ready in %.1f ms\n", (esp_timer_get_time() - t) / 1000.0);
    t = esp_timer_get_time();
    if (!pdata || tvp_init(pdata, psize)) printf("phonemizer not available: %s\n", pdata ? tvp_error() : "no partition");
    else printf("phonemizer ready in %.1f ms\n", (esp_timer_get_time() - t) / 1000.0);
    printf("board: %s\n", board_name());
    if (board_init(50) == 0) board_beep(880, 120);
    printf("free PSRAM %u (largest block %u), internal %u\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM), (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    have_test = tdata && !tv_file_open(&test, tdata, tsize);
    printf("\nType a phrase and press Enter. Commands: voice F2|M2, vol N, beep, look N, prof, TEST\n");
    static char line[8192];
    for (;;) {
        printf("READY\ntts> ");
        fflush(stdout);
        if (!read_line(line, sizeof line - 1)) continue;        // one byte left for say()
        if (!strncmp(line, "SYN ", 4)) run_syn(line);
        else if (!strcmp(line, "TEST")) run_test();
        else if (!strcmp(line, "beep")) board_beep(880, 300);
        else if (!strcmp(line, "mmbench")) mmbench();
        else if (!strcmp(line, "prof")) { profiling = !profiling; printf("profiling %s\n", profiling ? "on" : "off"); }
        else if (!strncmp(line, "voice ", 6)) {
            if (!strcmp(line + 6, "F2") || !strcmp(line + 6, "M2")) voice_id = line[6] == 'M';
            printf("voice %s\n", voice_id ? "M2" : "F2");
        }
        else if (!strncmp(line, "look ", 5)) {     // the memory budget assumes a small look-ahead
            int v = atoi(line + 5);
            acoustic_lookahead = v < 0 ? 0 : v > 64 ? 64 : v;
            printf("acoustic look-ahead %d\n", acoustic_lookahead);
        }
        else if (!strncmp(line, "vol ", 4)) {
            int v = atoi(line + 4);
            printf(board_set_volume(v) ? "volume: codec not available\n" : "volume %d%%\n", v < 0 ? 0 : v > 100 ? 100 : v);
        }
        else if (!strncmp(line, "PHON ", 5)) {     // debugging: phonemes only
            int64_t tp = esp_timer_get_time();
            int n = tvp_phonemize(line + 5, tok_ids, tok_stress, MAX_TOKENS);
            printf("PHON %d in %.1f ms:", n, (esp_timer_get_time() - tp) / 1000.0);
            for (int i = 0; i < n; i++) printf(" %d", (int)tok_ids[i]);
            if (n < 0) printf(" %s", tvp_error());
            printf("\n");
        }
        else say(line);
        fflush(stdout);
    }
done:
    printf("tv: finished\n");
    vTaskDelete(NULL);
}

void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(1500));    // let the USB console attach
    usb_serial_jtag_driver_config_t cfg = {.tx_buffer_size = 4096, .rx_buffer_size = 8192};
    usb_serial_jtag_driver_install(&cfg);
    usb_serial_jtag_vfs_use_driver();
    usb_serial_jtag_vfs_set_rx_line_endings(ESP_LINE_ENDINGS_LF);
    setvbuf(stdin, NULL, _IONBF, 0);
    printf("\n#### tiny-voca firmware ####\n");
    xTaskCreatePinnedToCore(run, "tv_run", 49152, NULL, 5, NULL, 0);
}
