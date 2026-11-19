// 8-bit register access on an I2C device (100 ms timeout).
#pragma once
#include <stdint.h>
#include "driver/i2c_master.h"

static inline esp_err_t i2c_reg_read(i2c_master_dev_handle_t dev, uint8_t reg, uint8_t *value) {
    return i2c_master_transmit_receive(dev, &reg, 1, value, 1, 100);
}

static inline esp_err_t i2c_reg_write(i2c_master_dev_handle_t dev, uint8_t reg, uint8_t value) {
    uint8_t buf[2] = {reg, value};
    return i2c_master_transmit(dev, buf, 2, 100);
}

// reg = (reg & keep) | set, written only when the read worked.
static inline esp_err_t i2c_reg_update(i2c_master_dev_handle_t dev, uint8_t reg, uint8_t keep, uint8_t set) {
    uint8_t v;
    esp_err_t err = i2c_reg_read(dev, reg, &v);
    return err ? err : i2c_reg_write(dev, reg, (uint8_t)((v & keep) | set));
}
