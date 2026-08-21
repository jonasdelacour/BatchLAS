// WP4 step 0.2 -- what does the runtime REPORT, and what actually LAUNCHES?
//
// Part 1: dump every SYCL device info field that bears on SLM sizing.
// Part 2: bisect the largest sycl::local_accessor<uint8_t,1> that (a) submits
//         without a synchronous or asynchronous exception AND (b) produces the
//         correct answer -- a kernel that "launches" but writes nothing would
//         be a false green.
//
// Build: ./build_probe.sh   (flags copied from the project's own link line)
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

// Touch every byte of the local accessor, then reduce it, so the allocation
// cannot be dead-code-eliminated and a silently-truncated allocation shows up
// as a WRONG ANSWER rather than a pass.
static bool try_slm(sycl::queue& q, size_t bytes, size_t wg_size, std::string& err) {
    err.clear();
    const size_t n_wg = 4;                       // >1 work-group: catches per-SM limits too
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
                        sycl::group_barrier(it.get_group());
                        int64_t part = 0;
                        for (size_t i = lid; i < bytes; i += nl) part += (int64_t)slm[i];
                        int64_t tot = sycl::reduce_over_group(it.get_group(), part, sycl::plus<int64_t>());
                        if (lid == 0) out[it.get_group(0)] = tot;
                    });
            });
            q.wait_and_throw();
        }                                          // buffer destructor copies back
        for (size_t i = 0; i < n_wg; ++i)
            if (host[i] != expect) {
                char b[256];
                std::snprintf(b, sizeof b, "WRONG ANSWER wg%zu got %lld want %lld",
                              i, (long long)host[i], (long long)expect);
                err = b;
                return false;
            }
        return true;
    } catch (sycl::exception const& e) {
        err = e.what();
        return false;
    } catch (std::exception const& e) {
        err = e.what();
        return false;
    }
}

int main(int argc, char** argv) {
    sycl::queue q{sycl::gpu_selector_v,
                  sycl::property_list{sycl::property::queue::in_order{}}};
    auto d = q.get_device();
    std::printf("# device                         : %s\n", d.get_info<sycl::info::device::name>().c_str());
    std::printf("# vendor                         : %s\n", d.get_info<sycl::info::device::vendor>().c_str());
    std::printf("# driver                         : %s\n", d.get_info<sycl::info::device::driver_version>().c_str());
    std::printf("# local_mem_size   (DeviceProperty::LOCAL_MEM_SIZE) : %zu\n",
                (size_t)d.get_info<sycl::info::device::local_mem_size>());
    std::printf("# local_mem_type                 : %d\n",
                (int)d.get_info<sycl::info::device::local_mem_type>());
    std::printf("# max_work_group_size            : %zu\n",
                (size_t)d.get_info<sycl::info::device::max_work_group_size>());
    std::printf("# max_compute_units              : %u\n",
                d.get_info<sycl::info::device::max_compute_units>());
    std::printf("# max_num_sub_groups             : %u\n",
                d.get_info<sycl::info::device::max_num_sub_groups>());
    std::printf("# sub_group_sizes                : ");
    for (auto s : d.get_info<sycl::info::device::sub_group_sizes>()) std::printf("%zu ", (size_t)s);
    std::printf("\n");

    std::vector<size_t> wgs;
    if (argc > 1) { for (int i = 1; i < argc; ++i) wgs.push_back((size_t)std::stoul(argv[i])); }
    else wgs = {32, 64, 128, 256, 512, 1024};

    for (size_t wg : wgs) {
        if (wg > (size_t)d.get_info<sycl::info::device::max_work_group_size>()) continue;
        std::string err;
        // Named spot checks first, then a bisection.
        for (size_t probe : {(size_t)45056, (size_t)49152, (size_t)65536, (size_t)71744,
                             (size_t)97280, (size_t)101376}) {
            bool ok = try_slm(q, probe, wg, err);
            std::printf("wg=%-5zu probe=%-8zu %s%s%s\n", wg, probe, ok ? "OK" : "FAIL",
                        ok ? "" : " : ", err.c_str());
            std::fflush(stdout);
        }
        // Bisect on [lo known-good, hi known-bad]. 1024 is assumed good; verify.
        size_t lo = 1024, hi = 200000;
        if (!try_slm(q, lo, wg, err)) { std::printf("wg=%-5zu BASELINE 1024 FAILED: %s\n", wg, err.c_str()); continue; }
        if (try_slm(q, hi, wg, err))  { std::printf("wg=%-5zu ceiling >= %zu\n", wg, hi); continue; }
        while (hi - lo > 1) {
            size_t mid = lo + (hi - lo) / 2;
            if (try_slm(q, mid, wg, err)) lo = mid; else hi = mid;
        }
        std::string e2; try_slm(q, hi, wg, e2);
        std::printf("wg=%-5zu CEILING = %zu bytes  (first failure at %zu: %s)\n",
                    wg, lo, hi, e2.c_str());
        std::fflush(stdout);
    }
    return 0;
}
