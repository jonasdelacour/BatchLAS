// The launch hole is order-dependent -- and that is the dangerous part.
//
// CUDA raises a kernel function's MaxDynamicSharedMemorySize attribute lazily, and
// the attribute is STICKY per function. potrf's CTA kernel is templated on
// <T, NB, TS, Scope> and takes n as a RUNTIME argument, so one function serves
// every n. If a large n runs first the attribute is already raised and a later
// n=110 launch succeeds; in a fresh process that starts at n=110 the same launch
// fails. That is an intermittent, order-dependent failure, not a deterministic one.
//
// 49064 B is exactly what float n=110 requests under section 4.1 + the W9 off[] term:
//   LDA=111 -> 111*110*4 + 16*4 + 64 + 4*ceil((110-16)/4) = 48840 + 64 + 64 + 96
//
// usage: slm_hole_order [--warm]      --warm launches 65536 B first
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

// One kernel NAME for every size -- this is the point of the test. A distinct
// template parameter per size would give distinct CUfunctions and hide the effect.
class HoleKernel;

static bool run(sycl::queue& q, size_t bytes, std::string& err) {
    err.clear();
    std::vector<int64_t> host(1, -1);
    int64_t expect = 0;
    for (size_t i = 0; i < bytes; ++i) expect += (int64_t)(uint8_t)((i * 31u + 7u) & 0xFF);
    try {
        {
            sycl::buffer<int64_t, 1> buf(host.data(), sycl::range<1>(1));
            q.submit([&](sycl::handler& cgh) {
                auto slm = sycl::local_accessor<uint8_t, 1>(sycl::range<1>(bytes), cgh);
                auto out = buf.get_access<sycl::access::mode::write>(cgh);
                cgh.parallel_for<HoleKernel>(
                    sycl::nd_range<1>(sycl::range<1>(128), sycl::range<1>(128)),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                        const size_t lid = it.get_local_id(0);
                        const size_t nl  = it.get_local_range(0);   // runtime stride:
                        // a literal 128 here crashes the DPC++ loop vectoriser
                        // (LoopVectorize.cpp:7244 assertion) in this toolchain.
                        for (size_t i = lid; i < bytes; i += nl)
                            slm[i] = (uint8_t)((i * 31u + 7u) & 0xFF);
                        sycl::group_barrier(it.get_group());
                        int64_t part = 0;
                        for (size_t i = lid; i < bytes; i += nl) part += (int64_t)slm[i];
                        int64_t tot = sycl::reduce_over_group(it.get_group(), part, sycl::plus<int64_t>());
                        if (lid == 0) out[0] = tot;
                    });
            });
            q.wait_and_throw();
        }
        if (host[0] != expect) { err = "WRONG ANSWER"; return false; }
        return true;
    } catch (sycl::exception const& e) { err = e.what(); return false; }
      catch (std::exception const& e) { err = e.what(); return false; }
}

int main(int argc, char** argv) {
    bool warm = argc > 1 && std::string(argv[1]) == "--warm";
    sycl::queue q{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    std::string e;
    if (warm) std::printf("warm 65536      : %s %s\n", run(q, 65536, e) ? "OK" : "FAIL", e.c_str());
    std::printf("float n=110 49064: %s %s\n", run(q, 49064, e) ? "OK" : "FAIL", e.c_str());
    std::printf("padded      49408: %s %s\n", run(q, 49408, e) ? "OK" : "FAIL", e.c_str());
    std::printf("float n=110 49064 (after pad): %s %s\n", run(q, 49064, e) ? "OK" : "FAIL", e.c_str());
    return 0;
}
