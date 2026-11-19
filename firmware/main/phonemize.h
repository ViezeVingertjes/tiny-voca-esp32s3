// C interface to the dizione phonemizer (C++), used by the firmware.
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int tvp_init(const void *container, size_t size);                       // 0 on success
// Text -> phoneme ids and stress. Returns the count (0: none, see tvp_error()), -2 when there are
// more than max, or -1 on another error (see tvp_error()).
int tvp_phonemize(const char *text, int32_t *ids, int32_t *stress, int max);
const char *tvp_error(void);
#ifdef __cplusplus
}
#endif
