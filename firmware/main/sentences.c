// Port of tiny_voca.infer.sentences/finished (ASCII rules; "..." as U+2026 is treated as a mark).
#include "sentences.h"
#include <ctype.h>
#include <string.h>
#include <strings.h>

static int is_ellipsis(const char *p, const char *start) {     // U+2026 ends at p
    return p - start >= 2 && (unsigned char)p[-2] == 0xE2 && (unsigned char)p[-1] == 0x80 && (unsigned char)p[0] == 0xA6;
}

static int is_mark(const char *p, const char *start) {
    return *p == '.' || *p == '!' || *p == '?' || is_ellipsis(p, start);
}

static int is_closing(char c) { return c == '"' || c == '\'' || c == ')' || c == ']'; }

// The segment [start, end) ends with an abbreviation or an initial followed by "."
static int not_an_end(const char *start, const char *end) {
    static const char *const abbrev[] = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "no",
                                         "inc", "ltd", "co", "e.g", "i.e", "a.m", "p.m"};
    if (end - start < 2 || end[-1] != '.') return 0;
    const char *w = end - 1;
    while (w > start && (isalpha((unsigned char)w[-1]) || w[-1] == '.')) w--;
    size_t len = (size_t)(end - 1 - w);
    if (len == 1 && isalpha((unsigned char)*w)) return 1;
    for (size_t i = 0; i < sizeof abbrev / sizeof *abbrev; i++)
        if (strlen(abbrev[i]) == len && !strncasecmp(w, abbrev[i], len)) return 1;
    return 0;
}

static int has_alnum(const char *s) {
    for (; *s; s++) if (isalnum((unsigned char)*s)) return 1;
    return 0;
}

static void collapse_spaces(char *text) {
    char *w = text;
    int space = 1;
    for (char *r = text; *r; r++) {
        if (isspace((unsigned char)*r)) { if (!space) *w++ = ' '; space = 1; }
        else { *w++ = *r; space = 0; }
    }
    if (w > text && w[-1] == ' ') w--;
    *w = 0;
}

int tv_split_sentences(char *text, char **out, int max) {
    collapse_spaces(text);
    int n = 0;
    char *seg = text;
    for (char *p = text; *p && n < max - 1; p++) {
        if (*p != ' ' || p == text) continue;
        const char *q = p - 1;
        if (is_closing(*q) && q > text) q--;
        if (!is_mark(q, text)) continue;
        const char *next = p + 1;
        if (*next == '"' || *next == '\'' || *next == '(' || *next == '[') next++;
        if (!isupper((unsigned char)*next) && !isdigit((unsigned char)*next)) continue;
        if (not_an_end(seg, p)) continue;
        *p = 0;
        if (has_alnum(seg)) out[n++] = seg;
        seg = p + 1;
    }
    if (*seg && has_alnum(seg) && n < max) out[n++] = seg;
    return n;
}

int tv_split_clauses(char *text, char **out, int max) {
    int n = 0;
    char *seg = text;
    for (char *p = text; *p && n < max - 1; p++) {
        if (*p != ' ' || p == text) continue;
        char c = p[-1];
        int dash = p - text >= 3 && (unsigned char)p[-3] == 0xE2 && (unsigned char)p[-2] == 0x80 && (unsigned char)p[-1] == 0x94;
        if (c != ',' && c != ';' && c != ':' && !dash) continue;
        *p = 0;
        if (has_alnum(seg)) out[n++] = seg;
        seg = p + 1;
    }
    if (*seg && has_alnum(seg) && n < max) out[n++] = seg;
    return n;
}

void tv_finish_text(char *text) {
    size_t len = strlen(text);
    while (len && isspace((unsigned char)text[len - 1])) text[--len] = 0;
    if (!len) return;
    const char *last = strrchr(text, ' ');
    last = last ? last + 1 : text;
    int alnum = has_alnum(last);
    size_t i = len;
    while (i > (size_t)(last - text) && strchr("\"')]", text[i - 1])) i--;     // closing marks after the word
    char end = i ? text[i - 1] : 0;
    int ellipsis = i >= 3 && is_ellipsis(text + i - 1, text);
    int marked = alnum ? (strchr(".!?,;:", end) != NULL && end) || ellipsis
                       : (strpbrk(last, ".!?") != NULL || !strcmp(last, ",") || !strcmp(last, ";") || !strcmp(last, ":"));
    if (!marked) { text[len] = '.'; text[len + 1] = 0; }
}
