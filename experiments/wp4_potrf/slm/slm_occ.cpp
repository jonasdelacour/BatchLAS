// ONE launch, one SLM size, one work-group size -- so that `ncu` reports exactly
// one kernel and its launch__occupancy_limit_* metrics can be read unambiguously.
//
// Caveat recorded up front: the register count of THIS kernel is not the register
// count of the real potrf CTA kernel, so launch__occupancy_limit_registers here is
// not predictive. launch__occupancy_limit_shared_mem IS, because it depends only on
// the shared-memory request and the carveout the driver picks for it.
//
// usage: slm_occ <bytes> <wg_size> [num_wg]
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    size_t bytes = argc > 1 ? std::stoul(argv[1]) : 45056;
    size_t wg    = argc > 2 ? std::stoul(argv[2]) : 128;
    size_t n_wg  = argc > 3 ? std::stoul(argv[3]) : 1024;
    sycl::queue q{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    std::vector<int64_t> host(n_wg, -1);
    {
        sycl::buffer<int64_t, 1> buf(host.data(), sycl::range<1>(n_wg));
        q.submit([&](sycl::handler& cgh) {
            auto slm = sycl::local_accessor<uint8_t, 1>(sycl::range<1>(bytes), cgh);
            auto out = buf.get_access<sycl::access::mode::write>(cgh);
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(n_wg * wg), sycl::range<1>(wg)),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                    const size_t lid = it.get_local_id(0);
                    const size_t nl  = it.get_local_range(0);
                    for (size_t i = lid; i < bytes; i += nl)
                        slm[i] = (uint8_t)((i * 31u + 7u) & 0xFF);
                    sycl::group_barrier(it.get_group());
                    int64_t part = 0;
                    for (size_t i = lid; i < bytes; i += nl) part += (int64_t)slm[i];
                    int64_t tot = sycl::reduce_over_group(it.get_group(), part, sycl::plus<int64_t>());
                    if (lid == 0) out[it.get_group(0)] = tot;
                });
        });
        q.wait_and_throw();
    }
    int64_t expect = 0;
    for (size_t i = 0; i < bytes; ++i) expect += (int64_t)(uint8_t)((i * 31u + 7u) & 0xFF);
    std::printf("bytes=%zu wg=%zu num_wg=%zu result=%s\n", bytes, wg, n_wg,
                host[0] == expect ? "CORRECT" : "WRONG");
    return host[0] == expect ? 0 : 1;
}
