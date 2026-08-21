// Second probe, no group collectives -- so the kernel should carry ZERO static
// shared memory. slm_probe measured a ceiling of 101120 = 101376 - 256, and a
// reproducible FAILURE HOLE at dynamic bytes in (48896, 49152]; both are explained
// if the kernel carries 256 B of static shared (reduce_over_group's cross-sub-group
// scratch: 8 B x 32 sub-groups) and the limit is on static + dynamic.
//
// If that model is right, this kernel's ceiling is 101376 exactly and its hole is
// empty. If the ceiling is still 101120, the 256 B belongs to the runtime, not to
// the reduction, and every potrf sizing formula must reserve it.
//
// usage: slm_probe_nostatic [wg_size ...]
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

static bool try_slm(sycl::queue& q, size_t bytes, size_t wg_size, std::string& err) {
    err.clear();
    const size_t n_wg = 4;
    std::vector<int64_t> host(n_wg, -1);
    int64_t expect = 0;
    for (size_t i = 0; i < bytes; ++i) expect += (int64_t)(uint8_t)((i * 31u + 7u) & 0xFF);
    try {
        {
            sycl::buffer<int64_t, 1> buf(host.data(), sycl::range<1>(n_wg));
            q.submit([&](sycl::handler& cgh) {
                auto slm = sycl::local_accessor<uint8_t, 1>(sycl::range<1>(bytes), cgh);
                auto out = buf.get_access<sycl::access::mode::write>(cgh);
                cgh.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(n_wg * wg_size), sycl::range<1>(wg_size)),
                    [=](sycl::nd_item<1> it) {
                        const size_t lid = it.get_local_id(0);
                        const size_t nl  = it.get_local_range(0);
                        for (size_t i = lid; i < bytes; i += nl)
                            slm[i] = (uint8_t)((i * 31u + 7u) & 0xFF);
                        sycl::group_barrier(it.get_group());   // bar.sync, no shared
                        if (lid == 0) {                        // serial sum, no collective
                            int64_t tot = 0;
                            for (size_t i = 0; i < bytes; ++i) tot += (int64_t)slm[i];
                            out[it.get_group(0)] = tot;
                        }
                    });
            });
            q.wait_and_throw();
        }
        for (size_t i = 0; i < n_wg; ++i)
            if (host[i] != expect) { err = "WRONG ANSWER"; return false; }
        return true;
    } catch (sycl::exception const& e) { err = e.what(); return false; }
      catch (std::exception const& e) { err = e.what(); return false; }
}

int main(int argc, char** argv) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    std::vector<size_t> wgs;
    if (argc > 1) for (int i = 1; i < argc; ++i) wgs.push_back((size_t)std::stoul(argv[i]));
    else wgs = {32, 128, 256};

    for (size_t wg : wgs) {
        std::string err;
        // Hole scan FIRST, before any launch big enough to trigger the opt-in --
        // the opt-in attribute is sticky per kernel function, so probing large
        // sizes first would hide the hole.
        size_t hole_lo = 0, hole_hi = 0;
        for (size_t b = 48800; b <= 49152; b += 8) {
            if (!try_slm(q, b, wg, err)) { if (!hole_lo) hole_lo = b; hole_hi = b; }
        }
        std::printf("wg=%-5zu hole in [48800,49152] : %s",
                    wg, hole_lo ? "" : "NONE\n");
        if (hole_lo) std::printf("[%zu,%zu]\n", hole_lo, hole_hi);
        std::fflush(stdout);

        size_t lo = 1024, hi = 200000;
        if (try_slm(q, hi, wg, err)) { std::printf("wg=%-5zu ceiling >= %zu\n", wg, hi); continue; }
        while (hi - lo > 1) {
            size_t mid = lo + (hi - lo) / 2;
            if (try_slm(q, mid, wg, err)) lo = mid; else hi = mid;
        }
        std::printf("wg=%-5zu CEILING = %zu bytes\n", wg, lo);
        std::fflush(stdout);
    }
    return 0;
}
