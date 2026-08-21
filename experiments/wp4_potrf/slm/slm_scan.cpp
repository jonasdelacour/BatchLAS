// Dense scan of local_accessor sizes -- exists to check whether the ceiling is a
// single cliff or whether there are HOLES (sizes that fail below the ceiling).
// The first run of slm_probe showed 49152 FAILING at wg=32 while 45056 and 65536
// both passed; a hole in the middle of the range would change what supports()
// may claim, so it is scanned rather than assumed to be a one-off.
//
// usage: slm_scan <wg_size> <lo> <hi> <step> [repeats]
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
                        sycl::group_barrier(it.get_group());
                        int64_t part = 0;
                        for (size_t i = lid; i < bytes; i += nl) part += (int64_t)slm[i];
                        int64_t tot = sycl::reduce_over_group(it.get_group(), part, sycl::plus<int64_t>());
                        if (lid == 0) out[it.get_group(0)] = tot;
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
    if (argc < 5) { std::fprintf(stderr, "usage: %s wg lo hi step [repeats]\n", argv[0]); return 2; }
    size_t wg   = std::stoul(argv[1]);
    size_t lo   = std::stoul(argv[2]);
    size_t hi   = std::stoul(argv[3]);
    size_t step = std::stoul(argv[4]);
    int reps    = argc > 5 ? std::stoi(argv[5]) : 1;
    sycl::queue q{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    std::printf("wg,bytes,rep,ok,err\n");
    for (size_t b = lo; b <= hi; b += step)
        for (int r = 0; r < reps; ++r) {
            std::string e;
            bool ok = try_slm(q, b, wg, e);
            for (auto& c : e) if (c == ',' || c == '\n') c = ' ';
            std::printf("%zu,%zu,%d,%d,%s\n", wg, b, r, ok ? 1 : 0, e.c_str());
            std::fflush(stdout);
        }
    return 0;
}
