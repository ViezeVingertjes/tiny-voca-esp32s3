cmake_minimum_required(VERSION 3.16)
# Microbenchmarks of the S3 (no model needed): idf.py build flash monitor, from this folder.
set(BASELINES OFF CACHE BOOL "Also run the fp32, memory bandwidth and libm benchmarks")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(tv_bench)
