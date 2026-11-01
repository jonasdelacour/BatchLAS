// SPDX-License-Identifier: MIT
// SASS lab driver: correctness of every variant against the hand kernel, then A/B timing.
// Usage: sass_lab [--batch=8192] [--warmup=3] [--iters=10] [--filter=op|prefix*] [--check-only]
#include <benchmarks/sass_lab/lab.hh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>

namespace lab {
std::vector<Entry>& registry() {
    static std::vector<Entry> r;
    return r;
}
void register_l1();
void register_l2a();
void register_l2b();
void register_l3();
void register_v2a();
void register_v2b();
}  // namespace lab

using namespace lab;

namespace {

template <class T> Data<T> make_data(sycl::queue& q, int batch, std::mt19937& rng) {
    Data<T> d;
    d.batch = batch;
    const std::size_t m = static_cast<std::size_t>(batch) * Data<T>::ms;
    const std::size_t v = static_cast<std::size_t>(batch) * Data<T>::vs;
    const std::size_t p = static_cast<std::size_t>(batch) * Data<T>::ps;
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    auto fill = [&](std::size_t count) {
        T* ptr = sycl::malloc_device<T>(count, q);
        std::vector<T> h(count);
        for (auto& e : h) e = static_cast<T>(u(rng));
        q.memcpy(ptr, h.data(), count * sizeof(T)).wait();
        return ptr;
    };
    d.A = fill(m); d.B = fill(m); d.Out = fill(m);
    d.X = fill(v); d.Y = fill(v);
    d.V = fill(p); d.W = fill(p);
    return d;
}

double ev_us(const sycl::event& e) {
    const auto s = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto t = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return static_cast<double>(t - s) * 1e-3;
}

template <class T> std::vector<T> snapshot(sycl::queue& q, const Data<T>& d, int items) {
    std::vector<T> h(static_cast<std::size_t>(items) * Data<T>::ms);
    q.memcpy(h.data(), d.Out, h.size() * sizeof(T)).wait();
    return h;
}

}  // namespace

int main(int argc, char** argv) {
    int batch = 8192, warmup = 3, iters = 10;
    std::string filter;
    bool check_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--batch=", 0) == 0) batch = std::stoi(a.substr(8));
        else if (a.rfind("--warmup=", 0) == 0) warmup = std::stoi(a.substr(9));
        else if (a.rfind("--iters=", 0) == 0) iters = std::stoi(a.substr(8));
        else if (a.rfind("--filter=", 0) == 0) filter = a.substr(9);
        else if (a == "--check-only") check_only = true;
    }
#ifndef LAB_ONLY_V2
    register_l1(); register_l2a(); register_l2b(); register_l3();
#endif
    register_v2a(); register_v2b();

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    std::fprintf(stderr, "device: %s batch=%d warmup=%d iters=%d\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str(), batch, warmup, iters);
    std::mt19937 rng(1234);
    Data<float> df = make_data<float>(q, batch, rng);
    Data<double> dd = make_data<double>(q, batch, rng);
    df.ld = dd.ld = 0;

    // Group entries by (op, type, N); each group is timed together, variants alternating.
    std::map<std::tuple<std::string, std::string, int>, std::vector<const Entry*>> groups;
    for (const auto& e : registry()) {
        if (!filter.empty() && e.op != filter && !(filter.back() == '*' && e.op.rfind(filter.substr(0, filter.size() - 1), 0) == 0))
            continue;
        groups[{e.op, e.type, e.N}].push_back(&e);
    }
    const int check_items = std::min(batch, 64);
    std::printf("op,type,N,n,variant,median_us,min_us,rel_to_hand,max_rel_err\n");
    for (auto& [key, entries] : groups) {
        const auto& [op, type, N] = key;
        std::vector<int> ns = N > 0 ? std::vector<int>{N} : std::vector<int>{32, 64};
        if (op == "gemm" && N == 0) ns = {32};
        for (int n : ns) {
            const bool is_f = type == "float";
            void* data = is_f ? static_cast<void*>(&df) : static_cast<void*>(&dd);
            df.ld = dd.ld = n;
            auto reset = [&] {
                if (is_f) q.memcpy(df.Out, df.A, sizeof(float) * batch * Data<float>::ms).wait();
                else q.memcpy(dd.Out, dd.A, sizeof(double) * batch * Data<double>::ms).wait();
            };
            // Correctness against the hand variant (first entry with V == kHand).
            std::vector<double> ref;
            std::vector<double> err(entries.size(), 0.0);
            for (std::size_t k = 0; k < entries.size(); ++k) {
                reset();
                entries[k]->run(q, data, n).wait();
                std::vector<double> got;
                if (is_f) { auto h = snapshot(q, df, check_items); got.assign(h.begin(), h.end()); }
                else { auto h = snapshot(q, dd, check_items); got.assign(h.begin(), h.end()); }
                if (entries[k]->V == kHand) ref = got;
                else if (!ref.empty()) {
                    double scale = 1.0, e = 0.0;
                    for (double r : ref) scale = std::max(scale, std::fabs(r));
                    for (std::size_t i = 0; i < got.size(); ++i) {
                        const double diff = std::fabs(got[i] - ref[i]);
                        e = std::max(e, std::isnan(diff) ? 1e300 : diff);
                    }
                    err[k] = e / scale;
                }
            }
            std::vector<std::vector<double>> t(entries.size());
            if (!check_only) {
                reset();
                for (int w = 0; w < warmup; ++w)
                    for (auto* e : entries) e->run(q, data, n).wait();
                for (int it = 0; it < iters; ++it) {
                    for (std::size_t j = 0; j < entries.size(); ++j) {
                        const std::size_t k = (it % 2 == 0) ? j : entries.size() - 1 - j;
                        if (entries[k]->inplace) reset();
                        auto ev = entries[k]->run(q, data, n);
                        ev.wait();
                        t[k].push_back(ev_us(ev));
                    }
                }
            }
            double hand_med = 0.0;
            std::vector<double> med(entries.size(), 0.0), mn(entries.size(), 0.0);
            for (std::size_t k = 0; k < entries.size(); ++k) {
                if (t[k].empty()) continue;
                auto s = t[k];
                std::sort(s.begin(), s.end());
                med[k] = s[s.size() / 2];
                mn[k] = s.front();
                if (entries[k]->V == kHand) hand_med = med[k];
            }
            for (std::size_t k = 0; k < entries.size(); ++k) {
                std::printf("%s,%s,%d,%d,%d,%.2f,%.2f,%.3f,%.2e\n", op.c_str(), type.c_str(), N, n, entries[k]->V,
                            med[k], mn[k], hand_med > 0 ? med[k] / hand_med : 0.0, err[k]);
            }
            std::fflush(stdout);
        }
    }
    std::fprintf(stderr, "LAB_DONE\n");
    return 0;
}
