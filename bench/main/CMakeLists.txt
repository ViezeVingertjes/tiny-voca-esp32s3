idf_component_register(SRCS "bench.c" "kernels.S" INCLUDE_DIRS ".")
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
