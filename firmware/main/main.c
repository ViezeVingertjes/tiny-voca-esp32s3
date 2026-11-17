// tiny-voca on the ESP32-S3: run the test vector from the `testvec` partition with the
// model from the `model` partition, report timings, and stream prosody + audio back over
// the console as base64 ("TVDUMP" blocks, decoded by esp32/tools/serial_dump.py).
#include <stdio.h>
#include <math.h>
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
#include "audio.h"
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

static tv_model model;
static tv_file test;
static int have_test;

static void run_test(void) {
    if (!have_test) { printf("no test vector\n"); return; }
    const tv_entry *ids_e = tv_file_find(&test, "ids"), *noise_e = tv_file_find(&test, "noise");
    int n = ids_e->shape[0];
    run_state st = {.golden_durations = tv_file_get(&test, "durations", TV_I32, n)};
    tv_options opt = {.speed = 1.0f, .noise = (const float *)(test.base + noise_e->offset), .debug = on_tensor, .debug_user = &st};
    tv_result r;
    tv_profile_reset();
    st.t0 = esp_timer_get_time();
    int rc = tv_synthesize(&model, tv_file_get(&test, "ids", TV_I32, n), tv_file_get(&test, "stress", TV_I32, n), n,
                           *(const int32_t *)tv_file_get(&test, "voice", TV_I32, 1), &opt, &r);
    double ms = (esp_timer_get_time() - st.t0) / 1000.0;
    if (rc) { printf("synthesize failed: %d\n", rc); return; }
    double audio_ms = 1000.0 * r.n_samples / TV_SAMPLE_RATE;
    tv_profile_report();
    printf("RESULT %d frames, %.0f ms of audio in %.0f ms (%.2fx real time), QACC saturations %ld\n", r.n_frames,
           audio_ms, ms, audio_ms / ms, r.qacc_saturations);
    if (st.prosody) dump_base64("prosody", st.prosody, (size_t)st.prosody_rows * 12, st.prosody_rows, 3);
    dump_base64("audio", r.audio, (size_t)r.n_samples * 4, 1, r.n_samples);
    tv_free(st.prosody);
    tv_free(r.audio); tv_free(r.durations);
    printf("TVDUMP_DONE\n");
}

// SYN <voice> <seed> <n> <id,id,...> <stress,stress,...>
static void run_syn(char *line) {
    int voice, n;
    unsigned seed;
    char *rest;
    if (sscanf(line, "SYN %d %u %d", &voice, &seed, &n) != 3 || n < 1 || n > 512) { printf("ERR bad command\n"); return; }
    rest = line;
    for (int i = 0; i < 4 && rest; i++) rest = strchr(rest + 1, ' ');
    int32_t *ids = tv_alloc(n * 4), *stress = tv_alloc(n * 4);
    char *tok = rest ? strtok(rest + 1, " ") : NULL;
    char *ids_s = tok, *str_s = tok ? strtok(NULL, " \r\n") : NULL;
    int ok = ids_s && str_s;
    for (int i = 0; ok && i < n; i++) {
        ids[i] = strtol(ids_s, &ids_s, 10); if (*ids_s == ',') ids_s++;
        stress[i] = strtol(str_s, &str_s, 10); if (*str_s == ',') str_s++;
        ok = ids[i] >= 0 && ids[i] < 76 && stress[i] >= 0 && stress[i] < 3;
    }
    if (!ok) { printf("ERR bad tokens\n"); tv_free(ids); tv_free(stress); return; }
    tv_options opt = {.speed = 1.0f, .seed = seed};
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

// Frames that fit in the free PSRAM: ~2.7 KB per frame at the peak (tiled stacks, streaming
// vocoder) plus ~0.7 MB of fixed tile and vocoder buffers; the largest single buffer is
// T x 768 bytes. The model itself handles at most 3000 frames.
static int frame_budget(void) {
    long total = (long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM) - 1200 * 1024;
    long largest = (long)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) - 64 * 1024;
    long frames = total / 2900;
    if (largest / 800 < frames) frames = largest / 800;
    if (frames > 3000) frames = 3000;
    return frames > 0 ? (int)frames : 0;
}

// Streams synthesized audio to the speaker in 0.2 s chunks (clipped at +-0.99).
typedef struct {
    int16_t *chunk;
    int fill;
    float peak;
} stream_state;

#define STREAM_CHUNK 4800

static void stream_flush(stream_state *st) {
    if (!st->fill) return;
    audio_play(st->chunk, st->fill);                 // the player frees it
    st->chunk = NULL;
    st->fill = 0;
}

static void on_audio(void *user, const float *x, int n) {
    stream_state *st = user;
    for (int i = 0; i < n; i++) {
        if (!st->chunk) st->chunk = tv_alloc(STREAM_CHUNK * 2);
        float v = x[i];
        if (fabsf(v) > st->peak) st->peak = fabsf(v);
        v = v > 0.99f ? 0.99f : v < -0.99f ? -0.99f : v;
        st->chunk[st->fill++] = (int16_t)lrintf(v * 32767.0f);
        if (st->fill == STREAM_CHUNK) stream_flush(st);
    }
}

// One piece of text: phonemize, synthesize, queue for playback. Splits at clauses when too long.
static int say_piece(char *text, int voice, unsigned seed, int depth) {
    static int32_t ids[512], stress[512];
    int n = tvp_phonemize(text, ids, stress, 512);
    if (n == -2 || n > 0) {
        stream_state st = {0};
        tv_options opt = {.speed = 1.0f, .seed = seed, .max_frames = frame_budget(), .on_audio = on_audio, .audio_user = &st};
        tv_result r;
        int64_t t0 = esp_timer_get_time();
        int rc = n > 0 ? tv_synthesize(&model, ids, stress, n, voice, &opt, &r) : TV_TOO_LONG;
        if (rc == 0) {
            stream_flush(&st);
            double ms = (esp_timer_get_time() - t0) / 1000.0, audio_ms = 1000.0 * r.n_samples / TV_SAMPLE_RATE;
            printf("  [%d phonemes, %d frames, %.2f s audio in %.2f s, RTF %.2f, peak %.2f] %s\n", n, r.n_frames,
                   audio_ms / 1000, ms / 1000, audio_ms / ms, st.peak, text);
            tv_free(r.durations);
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
        printf("  too long for memory (%d frames, budget %d): %s\n", r.n_frames, opt.max_frames, text);
        return -1;
    }
    printf("  cannot phonemize: %s (%s)\n", tvp_error(), text);
    return -1;
}

static int voice_id = 1;      // M2

static void say(char *line) {
    static char buf[2048];
    snprintf(buf, sizeof buf - 2, "%s", line);
    tv_finish_text(buf);
    char *parts[64];
    int k = tv_split_sentences(buf, parts, 64);
    for (int i = 0; i < k; i++) say_piece(parts[i], voice_id, (unsigned)i, 0);
}

// Line input with echo and backspace (works in idf.py monitor and from scripts).
static int read_line(char *buf, int size) {
    int n = 0;
    for (;;) {
        int c = fgetc(stdin);
        if (c == EOF) { vTaskDelay(pdMS_TO_TICKS(10)); continue; }
        if (c == '\r' || c == '\n') {
            if (n == 0) continue;           // second half of CRLF, or an empty line
            putchar('\n'); fflush(stdout);
            buf[n] = 0;
            return n;
        }
        if (c == 0x7f || c == 0x08) {
            if (n) { n--; fputs("\b \b", stdout); fflush(stdout); }
            continue;
        }
        if (n < size - 1 && (c >= 0x20 || c >= 0x80)) {
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
    if (audio_init(70) == 0) audio_beep(880, 120);
    printf("free PSRAM %u (largest block %u), internal %u\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM), (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    have_test = tdata && !tv_file_open(&test, tdata, tsize);
    printf("\nType a phrase and press Enter. Commands: voice F2|M2, vol N, beep, TEST\n");
    static char line[8192];
    for (;;) {
        printf("READY\ntts> ");
        fflush(stdout);
        read_line(line, sizeof line);
        if (!strncmp(line, "SYN ", 4)) run_syn(line);
        else if (!strcmp(line, "TEST")) run_test();
        else if (!strcmp(line, "beep")) audio_beep(880, 300);
        else if (!strncmp(line, "voice ", 6)) { voice_id = !strcmp(line + 6, "F2") ? 0 : 1; printf("voice %s\n", voice_id ? "M2" : "F2"); }
        else if (!strncmp(line, "vol ", 4)) printf("volume: re-run audio init not supported yet\n");
        else if (!strncmp(line, "PHON ", 5)) {     // debugging: phonemes only
            static int32_t ids[512], st[512];
            int n = tvp_phonemize(line + 5, ids, st, 512);
            printf("PHON %d:", n);
            for (int i = 0; i < n; i++) printf(" %d", (int)ids[i]);
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
