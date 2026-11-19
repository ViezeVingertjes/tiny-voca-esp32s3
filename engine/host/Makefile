# Host build of the engine (reference kernels, QACC emulation) for validation against the golden model.
# -ffp-contract=off: no fused multiply-adds, so every host computes the same floats. (The S3 build
# fuses them, so board output is checked against board output, see AGENTS.md.)
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter
CFLAGS += -ffp-contract=off
SRCS = main.c ../tv.c ../tv_dsp.c ../tv_kernels_ref.c
tvhost: $(SRCS) ../tv.h ../tv_kernels.h
	$(CC) $(CFLAGS) -o $@ $(SRCS) -lm
clean:
	rm -f tvhost
.PHONY: clean
