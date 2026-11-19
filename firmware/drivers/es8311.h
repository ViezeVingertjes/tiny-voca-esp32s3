// ES8311 mono codec, DAC path only: I2S slave, MCLK = 256 x fs from the MCLK pin, 16-bit samples.
#pragma once
#include "driver/i2c_master.h"
#include "i2s_player.h"

#define ES8311_ADDR 0x18            // on the Waveshare boards

// dev: the codec on an I2C bus. MCLK must already run. Sample rate fixed at 24 kHz from 6.144 MHz MCLK.
// Returns 0, -1 (no answer), -2 (not an ES8311) or -3 (a register access failed).
int es8311_start(i2c_master_dev_handle_t dev, int volume);
int es8311_set_volume(i2c_master_dev_handle_t dev, int volume);     // 0..100

// The audio path the boards share: I2C bus 0 on sda/scl (internal pull-ups) with the codec at
// ES8311_ADDR, the I2S player at 24 kHz on `i2s` (MCLK must run before the codec is set up), then
// the codec. Returns 0, -1 (I2C), -2 (I2S) or -3 (codec; *codec is then NULL, *bus stays usable).
int es8311_audio_start(int sda, int scl, const i2s_player_pins *i2s, int volume, i2c_master_bus_handle_t *bus,
                       i2c_master_dev_handle_t *codec);
