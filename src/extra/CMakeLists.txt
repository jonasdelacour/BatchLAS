set(EXTENSIONS_SOURCES
    norm.cc
    cond.cc
    transpose.cc
)

target_sources(batchlas PRIVATE ${EXTENSIONS_SOURCES})
