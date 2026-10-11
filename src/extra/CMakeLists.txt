set(EXTENSIONS_SOURCES
    norm.cc
    cond.cc
    random_cond.cc
    transpose.cc
    elementwise.cc
    matrix_checks.cc
)

target_sources(batchlas_extra_obj PRIVATE ${EXTENSIONS_SOURCES})
