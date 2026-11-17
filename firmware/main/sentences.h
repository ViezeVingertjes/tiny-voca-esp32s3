// Text splitting, ported from tiny_voca.infer: sentences() and finished(), plus clause splitting.
#pragma once
// Splits `text` (modified in place) into sentences; returns the count, pointers in `out`.
int tv_split_sentences(char *text, char **out, int max);
// Splits one sentence (in place) at whitespace after , ; : or an em dash.
int tv_split_clauses(char *text, char **out, int max);
// Appends "." when the text ends without a mark (buffer must have room for one byte).
void tv_finish_text(char *text);
