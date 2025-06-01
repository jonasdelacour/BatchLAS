set(EXTENSIONS_SOURCES
    norm.cc
    cond.cc
)

target_sources(batchlas PRIVATE ${EXTENSIONS_SOURCES})