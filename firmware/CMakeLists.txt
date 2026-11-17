cmake_minimum_required(VERSION 3.16)
# The board: a folder in boards/ (pins and glue in board.c, flash/PSRAM in sdkconfig.defaults,
# partitions.csv). Choose with: idf.py -B build/<board> -DBOARD=<board> build
set(BOARD "waveshare_s3_audio" CACHE STRING "Board folder in boards/")
if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/boards/${BOARD}/board.c")
    message(FATAL_ERROR "Unknown board '${BOARD}': no boards/${BOARD}/board.c")
endif()
set(SDKCONFIG_DEFAULTS "${CMAKE_CURRENT_LIST_DIR}/sdkconfig.defaults;${CMAKE_CURRENT_LIST_DIR}/boards/${BOARD}/sdkconfig.defaults")
set(SDKCONFIG "${CMAKE_CURRENT_LIST_DIR}/sdkconfig.${BOARD}")    # one configuration per board
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(tv_fw)
