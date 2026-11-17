idf_component_register(
    SRCS "main.c" "tv_kernels_s3.c" "kernels_s3.S" "../../engine/tv.c" "../../engine/tv_dsp.c"
    INCLUDE_DIRS "." "../../engine"
    REQUIRES esp_partition spi_flash esp_timer esp_driver_usb_serial_jtag)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -Wno-unused-parameter -Wno-format-truncation)
