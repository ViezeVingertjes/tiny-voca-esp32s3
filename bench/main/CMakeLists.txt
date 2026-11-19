idf_component_register(SRCS "bench.c" "kernels.S" INCLUDE_DIRS ".")
if(BASELINES)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE RUN_BASELINES=1)
endif()
