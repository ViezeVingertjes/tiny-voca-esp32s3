# Host build of the engine (reference kernels, QACC emulation) for validation against the golden model.
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter
tvhost: main.c ../tv.c ../tv_dsp.c ../tv_kernels_ref.c ../tv.h ../tv_kernels.h
	$(CC) $(CFLAGS) -o $@ main.c ../tv.c ../tv_dsp.c ../tv_kernels_ref.c -lm
clean:
	rm -f tvhost
