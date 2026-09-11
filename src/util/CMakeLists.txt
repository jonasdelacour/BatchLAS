set(UTIL_SOURCES
    queue-impl.cc
    # The only translation unit in the library that calls std::getenv. Its home
    # is here rather than in src/core because <batchlas/util/env.hh> -- the read
    # side it shares a contract with -- is a util header, and because the Queue
    # constructor that latches configure() shut lives next door in queue-impl.cc.
    settings.cc
    sycl-util-impl.cc
)

target_sources(batchlas_util_obj PRIVATE ${UTIL_SOURCES})

# Add specific SYCL compiler flags for these files
#set_source_files_properties(
#    queue-impl.cc
#    sycl-util-impl.cc
#    PROPERTIES COMPILE_FLAGS "-fsycl"
#)