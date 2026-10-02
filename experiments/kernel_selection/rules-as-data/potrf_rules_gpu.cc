// GPU driver for the rules-as-data prototype: the old facade and the new engine in ONE process.
//
//   potrf_rules_gpu equiv <cells.txt>              "<dtype> <n> <batch> <L|U>" per line; prints
//                                                  today's backend::potrf_route and the engine's
//                                                  pick, vendor present and vendor-free
//   potrf_rules_gpu run <dtype> <L|U> <n> <batch>  factor with both paths, residuals, sizes
//   potrf_rules_gpu size <dtype> <L|U> <n> <batch> potrf_buffer_size old vs new
//
// Honours BATCHLAS_ROUTING_PROFILE and BATCHLAS_POTRF_ROUTE in both paths.

#include "../../../src/routing/potrf_tiers.hh"

#include <batchlas/util/sycl-vector.hh>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace batchlas;
namespace rp = batchlas::routing::potrf;
constexpr Backend kB = Backend::CUDA;

namespace {

std::string name_of(dispatch::Route r) {
    if (dispatch::is_vendor(r)) return "vendor";
    switch (r.algo) {
        case dispatch::Algorithm::Tiny: return "native:tiny";
        case dispatch::Algorithm::CTA: return "native:cta";
        case dispatch::Algorithm::LPanel: return "native:lpanel";
        case dispatch::Algorithm::Blocked: return "native:blocked";
        default: return "?";
    }
}

template <class T>
std::string new_pick(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo u, bool vendor,
                     std::string* why = nullptr) {
    try {
        const auto s = rp::potrf_select<kB, T>(q, A, u, vendor, false);
        if (why) {
            char b[96];
            std::snprintf(b, sizeof b, "%s R%04u %s/%s", std::string(to_string(s.reason)).c_str(),
                          s.rule_id, std::string(s.rules->arch).c_str(),
                          std::string(to_string(s.source)).c_str());
            *why = b;
        }
        return std::string(s.id());
    } catch (const routing::no_route_error&) {
        if (why) *why = "no legal tier";
        return "vendor";
    }
}

template <class T>
void equiv_one(Queue& q, int n, long long batch, Uplo u, const char* dt, int& cells, int& bad) {
    const MatrixView<T, MatrixFormat::Dense> A(nullptr, n, n, n, n * n, static_cast<int>(batch));
    for (bool vendor : {true, false}) {
        const std::string old = name_of(backend::potrf_route<kB, T>(q, A, u, vendor));
        std::string why;
        const std::string neu = new_pick<T>(q, A, u, vendor, &why);
        ++cells;
        const bool ok = old == neu;
        bad += !ok;
        std::printf("%s %s %c %d %lld vendor=%d today=%s rules=%s [%s]\n", ok ? "OK  " : "DIFF", dt,
                    u == Uplo::Upper ? 'U' : 'L', n, batch, vendor, old.c_str(), neu.c_str(),
                    why.c_str());
    }
}

template <class T>
T make(double re, double im) {
    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        return T(static_cast<typename T::value_type>(re), static_cast<typename T::value_type>(im));
    } else {
        return T(re);
    }
}

template <class T>
double absd(T v) { return std::abs(v); }

template <class T>
T conjv(T v) {
    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        return std::conj(v);
    } else {
        return v;
    }
}

// Hermitian, diagonally dominant, item-dependent, nonzero imaginary off-diagonals.
template <class T>
T entry(int r, int c, int b, int n) {
    if (r == c) return make<T>(n + 1.0 + 0.01 * (b % 5), 0);
    const double s = 0.5 / (1.0 + std::abs(r - c)) * (1.0 + 0.001 * (b % 7));
    const double im = 0.1 * (r > c ? 1 : -1) / (1.0 + std::abs(r - c));
    return make<T>(s, im);
}

// || F F^H - A || / || A || over the referenced triangle; Lower: F = L, Upper: A = U^H U.
template <class T>
double residual(const T* f, int n, int ld, int b, Uplo u) {
    double num = 0, den = 0;
    for (int c = 0; c < n; ++c) {
        for (int r = c; r < n; ++r) {
            T s = make<T>(0, 0);
            if (u == Uplo::Lower) {
                for (int k = 0; k <= c; ++k) s += f[k * ld + r] * conjv(f[k * ld + c]);
            } else {
                for (int k = 0; k <= c; ++k) s += conjv(f[c * ld + k]) * f[r * ld + k];
                s = conjv(s);
            }
            const T a = entry<T>(r, c, b, n);
            num += absd(s - a) * absd(s - a);
            den += absd(a) * absd(a);
        }
    }
    return std::sqrt(num / den);
}

template <class T>
int run(Uplo u, int n, int batch, bool do_run) {
    Queue q(Device("gpu"));
    const int ld = n + 3;   // a non-natural ld
    const std::size_t sa = static_cast<std::size_t>(ld) * n;
    UnifiedVector<T> A1(sa * batch), A2(sa * batch);
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < n; ++c) {
            for (int r = 0; r < ld; ++r) {
                const T v = (r < n) ? entry<T>(r, c, b, n) : make<T>(1e30, 0);
                A1[b * sa + static_cast<std::size_t>(c) * ld + r] = v;
                A2[b * sa + static_cast<std::size_t>(c) * ld + r] = v;
            }
        }
    }
    UnifiedVector<T*> p1(batch), p2(batch);
    MatrixView<T, MatrixFormat::Dense> V1(A1.data(), n, n, ld, static_cast<int>(sa), batch, p1.data());
    MatrixView<T, MatrixFormat::Dense> V2(A2.data(), n, n, ld, static_cast<int>(sa), batch, p2.data());
    const std::size_t old_ws = potrf_buffer_size<kB, T>(q, V1, u);
    const auto sel = rp::potrf_select<kB, T>(q, V2, u);
    const std::size_t new_ws = rp::potrf_rules_buffer_size<kB, T>(q, V2, u);
    std::printf("SIZE n=%d batch=%d uplo=%c old=%zu new=%zu tier=%s reason=%s rule=R%04u (%s/%s)\n",
                n, batch, u == Uplo::Upper ? 'U' : 'L', old_ws, new_ws,
                std::string(sel.id()).c_str(), std::string(to_string(sel.reason)).c_str(),
                sel.rule_id, std::string(sel.rules->arch).c_str(),
                std::string(to_string(sel.source)).c_str());
    if (!do_run) return 0;
    const std::string old_route = name_of(backend::potrf_route<kB, T>(q, V1, u, true));
    UnifiedVector<std::byte> w1(old_ws ? old_ws : 1), w2(new_ws ? new_ws : 1);
    UnifiedVector<int32_t> i1(batch), i2(batch);
    (void)potrf<kB, T>(q, V1, u, w1.to_span(), i1.to_span());
    q.wait();
    auto t0 = std::chrono::steady_clock::now();
    (void)rp::potrf_rules<kB, T>(q, V2, u, w2.to_span(), i2.to_span());
    q.wait();
    auto t1 = std::chrono::steady_clock::now();
    double diff = 0, maxv = 0;
    int info_bad = 0;
    for (int b = 0; b < batch; ++b) {
        info_bad += (i1[b] != 0) + (i2[b] != 0);
        for (int c = 0; c < n; ++c) {
            for (int r = 0; r < n; ++r) {
                const bool ref = (u == Uplo::Lower) ? r >= c : r <= c;
                if (!ref) continue;
                const std::size_t k = b * sa + static_cast<std::size_t>(c) * ld + r;
                diff = std::max(diff, absd(A1[k] - A2[k]));
                maxv = std::max(maxv, absd(A1[k]));
            }
        }
    }
    const double r0 = residual<T>(A2.data(), n, ld, 0, u);
    const double rl = residual<T>(A2.data() + (batch - 1) * sa, n, ld, batch - 1, u);
    std::printf("RUN old_route=%s new_tier=%s residual(item0)=%.3e residual(last)=%.3e "
                "max|old-new|/max|old|=%.3e info_nonzero=%d new_ms=%.3f\n",
                old_route.c_str(), std::string(sel.id()).c_str(), r0, rl, diff / maxv, info_bad,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
    return 0;
}

template <class F>
int by_type(const std::string& t, F&& f) {
    if (t == "float") return f(float{});
    if (t == "double") return f(double{});
    if (t == "cfloat") return f(std::complex<float>{});
    if (t == "cdouble") return f(std::complex<double>{});
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    const std::string mode = argv[1];
    if (mode == "equiv") {
        Queue q(Device("gpu"));
        std::ifstream in(argv[2]);
        std::string line;
        int cells = 0, bad = 0;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string dt, ul;
            int n = 0;
            long long b = 0;
            if (!(ls >> dt >> n >> b >> ul)) continue;
            const Uplo u = ul == "U" ? Uplo::Upper : Uplo::Lower;
            by_type(dt, [&](auto z) {
                equiv_one<decltype(z)>(q, n, b, u, dt.c_str(), cells, bad);
                return 0;
            });
        }
        std::printf("SUMMARY %d decisions, %d differ\n", cells, bad);
        return bad ? 1 : 0;
    }
    if (argc < 6) return 2;
    const Uplo u = std::string(argv[3]) == "U" ? Uplo::Upper : Uplo::Lower;
    const int n = std::atoi(argv[4]), b = std::atoi(argv[5]);
    return by_type(argv[2], [&](auto z) { return run<decltype(z)>(u, n, b, mode == "run"); });
}
