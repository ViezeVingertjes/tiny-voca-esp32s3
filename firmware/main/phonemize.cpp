// C interface to the dizione C++ phonemizer, reading its data from the `phonemizer` partition.
#include "phonemize.h"
#include "dizione.h"
#include "tv.h"
#include "tv_kernels.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

static std::unique_ptr<dizione::Phonemizer> g_phonemizer;
static char g_error[160];

static std::string_view blob(const tv_file &f, const char *name) {
    const tv_entry *e = tv_file_find(&f, name);
    if (!e) return {};
    return std::string_view(reinterpret_cast<const char *>(f.base + e->offset), e->nbytes);
}

extern "C" int tvp_init(const void *container, size_t size) {
    tv_file f;
    if (tv_file_open(&f, container, size)) { snprintf(g_error, sizeof g_error, "phonemizer partition is not a TVQ1 file"); return -1; }
    try {
        dizione::Data data;
        data.lexicon_view = blob(f, "lexicon.tsv");                 // stays in mapped flash
        data.align = std::string(blob(f, "align.json"));
        data.g2p = std::string(blob(f, "g2p.json"));
        data.homographs = std::string(blob(f, "homographs.json"));
        data.g2p_weights = dizione::parse_npz(std::string(blob(f, "g2p.npz")));
        if (data.lexicon_view.empty() || data.align.empty()) { snprintf(g_error, sizeof g_error, "phonemizer data incomplete"); return -1; }
        dizione::g2p_parallel = tv_parallel2;                       // GRU directions on both cores
        g_phonemizer = std::make_unique<dizione::Phonemizer>(std::move(data));
    } catch (const std::exception &e) {
        snprintf(g_error, sizeof g_error, "%s", e.what());
        return -1;
    }
    return 0;
}

extern "C" int tvp_phonemize(const char *text, int32_t *ids, int32_t *stress, int max) {
    if (!g_phonemizer) { snprintf(g_error, sizeof g_error, "phonemizer not initialized"); return -1; }
    try {
        dizione::Phonemized p = g_phonemizer->phonemize(text);
        int n = (int)p.ids.size();
        if (n > max) { snprintf(g_error, sizeof g_error, "too many phonemes (%d > %d)", n, max); return -2; }
        for (int i = 0; i < n; i++) { ids[i] = p.ids[i]; stress[i] = p.stress[i]; }
        return n;
    } catch (const std::exception &e) {
        snprintf(g_error, sizeof g_error, "%s", e.what());
        return -1;
    }
}

extern "C" const char *tvp_error(void) { return g_error; }
