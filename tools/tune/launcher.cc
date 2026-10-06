// batchlas_tune: the SYCL-free front end of batchlas_tune_impl (docs/design/flat-kernel-selection.md
// §6). Linking libbatchlas enumerates devices during static init, which retains a CUDA primary
// context on every visible GPU, so the driver must never see a GPU: for driver modes this hides
// them all (CUDA_VISIBLE_DEVICES="") and passes the caller's value on in BATCHLAS_TUNE_PARENT_CVD,
// which the driver checks --devices against. Children (--cell, --info) pass through untouched.
// Deliberately links nothing but libc and libstdc++.

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <unistd.h>

int main(int argc, char** argv) {
    char self[PATH_MAX];
    const ssize_t len = ::readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (len <= 0) {
        std::perror("batchlas_tune: /proc/self/exe");
        return 2;
    }
    self[len] = '\0';
    std::string impl(self);
    impl = impl.substr(0, impl.rfind('/') + 1) + "batchlas_tune_impl";
    const bool child = argc > 1 && (!std::strcmp(argv[1], "--cell") || !std::strcmp(argv[1], "--info"));
    if (!child) {
        if (const char* v = std::getenv("CUDA_VISIBLE_DEVICES")) ::setenv("BATCHLAS_TUNE_PARENT_CVD", v, 1);
        else ::unsetenv("BATCHLAS_TUNE_PARENT_CVD");
        ::setenv("CUDA_VISIBLE_DEVICES", "", 1);
        ::setenv("BATCHLAS_TUNE_LAUNCHED", "1", 1);
    }
    ::execv(impl.c_str(), argv);
    std::fprintf(stderr, "batchlas_tune: cannot exec %s: %s\n", impl.c_str(), std::strerror(errno));
    return 127;
}
