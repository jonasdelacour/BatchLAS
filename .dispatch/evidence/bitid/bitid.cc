// Bit-identity harness: one binary, run against two trees' .so files via LD_LIBRARY_PATH.
// Each config prints: <op> <type> <tag> <params> <array> <fnv1a64 of raw bytes (incl. padding)>.
// Routes/env knobs come from the driver's environment; <tag> is argv's tag=... (printed verbatim).
#include <batchlas.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/internal/ormqr_blocked.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <src/ops/geqrf/geqrf_native.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace batchlas;
constexpr Backend BK = Backend::CUDA;
template <class T> using MV = MatrixView<T, MatrixFormat::Dense>;
template <class T> using R = typename base_type<T>::type;

static std::string g_tag = "-";
static std::map<std::string, std::string> g_args;
static Queue* g_q = nullptr;
static int g_batch = 1024;

static uint64_t fnv(const void* p, size_t n) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

template <class T> constexpr bool is_cx = !std::is_same_v<T, R<T>>;
template <class T> const char* tname() {
    if constexpr (std::is_same_v<T, float>) return "f";
    else if constexpr (std::is_same_v<T, double>) return "d";
    else if constexpr (std::is_same_v<T, std::complex<float>>) return "cf";
    else return "cd";
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ULL + 12345) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    double uni() { return (double)(next() >> 11) * (2.0 / 9007199254740992.0) - 1.0; }  // [-1,1)
};

template <class T> T rnd(Rng& g) {
    if constexpr (is_cx<T>) {
        double re = g.uni(), im = g.uni();
        if (std::fabs(im) < 1e-3) im = 0.25;
        return T(R<T>(re), R<T>(im));
    } else {
        return T(g.uni());
    }
}
template <class T> T poison() {
    const R<T> nan = std::numeric_limits<R<T>>::quiet_NaN();
    if constexpr (is_cx<T>) return T(nan, nan); else return nan;
}
template <class T> T conj_(T x) { if constexpr (is_cx<T>) return std::conj(x); else return x; }

// Padded batched matrix: ld = rows + 3, stride = ld * cols + 7, everything poisoned first.
template <class T> struct Buf {
    int r = 0, c = 0, ld = 0, stride = 0, batch = 0;
    UnifiedVector<T> v;
    Buf(int r_, int c_, int b_) : r(r_), c(c_), ld(r_ + 3), stride((r_ + 3) * c_ + 7), batch(b_),
        v(std::max<size_t>(1, (size_t)((r_ + 3) * c_ + 7) * b_), poison<T>()) {}
    T& at(int i, int j, int b) { return v[(size_t)b * stride + (size_t)j * ld + i]; }
    MV<T> view() { return MV<T>(v.data(), r, c, ld, stride, batch); }
    uint64_t hash() const { return fnv(v.data(), v.size() * sizeof(T)); }
};
// Flat per-item array with 7 poisoned trailing entries.
template <class T> struct Flat {
    int len, batch;
    UnifiedVector<T> v;
    Flat(int l, int b, T init) : len(l), batch(b), v((size_t)l * b + 7, init) {
        for (size_t i = (size_t)l * b; i < v.size(); ++i) v[i] = init;
    }
    Span<T> span() { return Span<T>(v.data(), (size_t)len * batch); }
    uint64_t hash() const { return fnv(v.data(), v.size() * sizeof(T)); }
};

static void line(const char* op, const char* ty, const std::string& params, const char* arr, uint64_t h) {
    std::printf("%s %s %s %s %s %016llx\n", op, ty, g_tag.c_str(), params.c_str(), arr, (unsigned long long)h);
}
static std::string sanitize(std::string s) {
    for (auto& ch : s) if (ch == ' ' || ch == '\n' || ch == '\t') ch = '_';
    if (s.size() > 120) s.resize(120);
    return s;
}
static void throw_line(const char* op, const char* ty, const std::string& params, const std::exception& e) {
    std::printf("%s %s %s %s THROW %s\n", op, ty, g_tag.c_str(), params.c_str(), sanitize(e.what()).c_str());
    try { g_q->wait(); } catch (...) {}
}

static std::vector<std::string> split(const std::string& s, char d = ',') {
    std::vector<std::string> out; std::string cur; std::stringstream ss(s);
    while (std::getline(ss, cur, d)) if (!cur.empty()) out.push_back(cur);
    return out;
}
static std::string arg(const std::string& k, const std::string& def) {
    auto it = g_args.find(k); return it == g_args.end() ? def : it->second;
}
static std::vector<int> ints(const std::string& k, const std::string& def) {
    std::vector<int> o; for (auto& s : split(arg(k, def))) o.push_back(std::stoi(s)); return o;
}
static std::vector<std::pair<int, int>> shapes(const std::string& k, const std::string& def) {
    std::vector<std::pair<int, int>> o;
    for (auto& s : split(arg(k, def))) { auto x = s.find('x'); o.push_back({std::stoi(s.substr(0, x)), std::stoi(s.substr(x + 1))}); }
    return o;
}
static bool want_type(const char* t) {
    for (auto& s : split(arg("types", "f,d,cf,cd"))) if (s == t) return true;
    return false;
}
static std::string P(std::initializer_list<std::pair<const char*, std::string>> kv) {
    std::string s;
    for (auto& [k, v] : kv) { if (!s.empty()) s += ","; s += k; s += "="; s += v; }
    return s;
}
static std::string I(long v) { return std::to_string(v); }
static const char* U(Uplo u) { return u == Uplo::Lower ? "L" : "U"; }

// Hermitian diagonally dominant (SPD) in the uplo triangle, Im(diag) != 0 (to be ignored),
// the other triangle independent junk. Items b % 7 == 3 indefinite at k = n/2 (info = k+1).
template <class T> void fill_hpd(Buf<T>& A, Uplo, uint64_t seed, bool indefinite) {
    const int n = A.r;
    for (int b = 0; b < A.batch; ++b) {
        Rng g(seed * 1000003 + b);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                T x = rnd<T>(g);
                if (i == j) {
                    R<T> d = R<T>(2 * n + 1) + R<T>(std::fabs((double)std::real(x)));
                    if (indefinite && b % 7 == 3 && i == n / 2) d = R<T>(-1);
                    if constexpr (is_cx<T>) A.at(i, i, b) = T(d, R<T>(0.37)); else A.at(i, i, b) = d;
                } else {
                    A.at(i, j, b) = x;  // junk in the unreferenced triangle; overwritten below if referenced
                }
            }
    }
}
// General random matrix; items b % 5 == 2 rank-deficient (column 1 = column 0), b % 5 == 4 rank-1 scaled.
template <class T> void fill_gen(Buf<T>& A, uint64_t seed, bool rankdef) {
    for (int b = 0; b < A.batch; ++b) {
        Rng g(seed * 1000003 + b);
        for (int j = 0; j < A.c; ++j)
            for (int i = 0; i < A.r; ++i) A.at(i, j, b) = rnd<T>(g);
        if (rankdef && A.c > 1 && b % 5 == 2)
            for (int i = 0; i < A.r; ++i) A.at(i, 1, b) = A.at(i, 0, b);
        if (rankdef && b % 5 == 4)
            for (int j = 1; j < A.c; ++j)
                for (int i = 0; i < A.r; ++i) A.at(i, j, b) = A.at(i, 0, b) * R<T>(j + 1) * R<T>(0.5);
    }
}
// Hermitian in uplo triangle (other triangle junk), Im(diag) != 0; b % 9 == 4 near-degenerate spectrum.
template <class T> void fill_herm(Buf<T>& A, uint64_t seed) {
    const int n = A.r;
    for (int b = 0; b < A.batch; ++b) {
        Rng g(seed * 1000003 + b);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                T x = rnd<T>(g);
                if (b % 9 == 4) x = (i == j) ? T(R<T>(1 + (i % 3))) : x * R<T>(1e-5);
                if (i == j) { if constexpr (is_cx<T>) x = T(std::real(x), R<T>(0.41)); }
                A.at(i, j, b) = x;
            }
    }
}

template <class T> void hash_info(const char* op, const std::string& p, UnifiedVector<int32_t>& info) {
    line(op, tname<T>(), p, "info", fnv(info.data(), info.size() * sizeof(int32_t)));
    long nz = 0; for (size_t i = 0; i < info.size(); ++i) nz += info[i] != 0;
    line(op, tname<T>(), p, "info_nonzero_count", (uint64_t)nz);
}

// ---------------------------------------------------------------- potrf
template <class T> void run_potrf() {
    for (int n : ints("n", "16")) for (auto us : split(arg("uplo", "L,U"))) {
        Uplo uplo = us == "L" ? Uplo::Lower : Uplo::Upper;
        std::string p = P({{"n", I(n)}, {"uplo", us}, {"batch", I(g_batch)}});
        try {
            Buf<T> A(n, n, g_batch); fill_hpd(A, uplo, 11 + n, true);
            UnifiedVector<int32_t> info(g_batch, -12345);
            size_t ws = potrf_buffer_size<BK, T>(*g_q, A.view(), uplo);
            UnifiedVector<std::byte> w(std::max<size_t>(ws, 1));
            potrf<BK, T>(*g_q, A.view(), uplo, Span<std::byte>(w.data(), ws), info.to_span());
            g_q->wait();
            line("potrf", tname<T>(), p, "A", A.hash());
            hash_info<T>("potrf", p, info);
        } catch (const std::exception& e) { throw_line("potrf", tname<T>(), p, e); }
    }
}
// ---------------------------------------------------------------- potri
// Synthetic Cholesky factor in the uplo triangle (real positive diagonal), junk in the other.
// Items b % 7 == 3 singular (zero diagonal at n/2).
template <class T> void run_potri() {
    for (int n : ints("n", "16")) for (auto us : split(arg("uplo", "L,U"))) for (int full : ints("full", "0,1")) {
        Uplo uplo = us == "L" ? Uplo::Lower : Uplo::Upper;
        std::string p = P({{"n", I(n)}, {"uplo", us}, {"full", I(full)}, {"batch", I(g_batch)}});
        try {
            Buf<T> A(n, n, g_batch);
            for (int b = 0; b < g_batch; ++b) {
                Rng g(77 + n * 31 + b);
                for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) {
                    T x = rnd<T>(g);
                    bool tri = uplo == Uplo::Lower ? i > j : i < j;
                    if (i == j) A.at(i, j, b) = T(R<T>(1.0 + 0.5 * std::fabs((double)std::real(x))));
                    else A.at(i, j, b) = tri ? x * R<T>(0.5 / std::sqrt((double)n)) : x;
                }
                if (b % 7 == 3) A.at(n / 2, n / 2, b) = T(0);
            }
            UnifiedVector<int32_t> info(g_batch, -12345);
            size_t ws = potri_buffer_size<BK, T>(*g_q, A.view(), uplo);
            UnifiedVector<std::byte> w(std::max<size_t>(ws, 1));
            potri<BK, T>(*g_q, A.view(), uplo, full != 0, Span<std::byte>(w.data(), ws), info.to_span());
            g_q->wait();
            line("potri", tname<T>(), p, "A", A.hash());
            hash_info<T>("potri", p, info);
        } catch (const std::exception& e) { throw_line("potri", tname<T>(), p, e); }
    }
}
// ---------------------------------------------------------------- geqrf (+ orgqr on its output)
template <class T> void orgqr_and_hash(const char* op, const std::string& p, Buf<T>& A, Flat<T>& tau) {
    try {
        size_t ws = orgqr_buffer_size<BK, T>(*g_q, A.view(), tau.span());
        UnifiedVector<std::byte> w(std::max<size_t>(ws, 1));
        orgqr<BK, T>(*g_q, A.view(), tau.span(), Span<std::byte>(w.data(), ws));
        g_q->wait();
        line(op, tname<T>(), p, "Q_orgqr", A.hash());
    } catch (const std::exception& e) { throw_line(op, tname<T>(), p + ",stage=orgqr", e); }
}
template <class T> void run_geqrf() {
    const bool do_orgqr = arg("orgqr", "1") == "1";
    for (auto [m, n] : shapes("shapes", "16x16")) {
        std::string p = P({{"m", I(m)}, {"n", I(n)}, {"batch", I(g_batch)}});
        try {
            Buf<T> A(m, n, g_batch); fill_gen(A, 21 + m * 7 + n, true);
            Flat<T> tau(std::min(m, n), g_batch, poison<T>());
            size_t ws = geqrf_buffer_size<BK, T>(*g_q, A.view(), tau.span());
            UnifiedVector<std::byte> w(std::max<size_t>(ws, 1));
            geqrf<BK, T>(*g_q, A.view(), tau.span(), Span<std::byte>(w.data(), ws));
            g_q->wait();
            line("geqrf", tname<T>(), p, "A", A.hash());
            line("geqrf", tname<T>(), p, "tau", tau.hash());
            if (do_orgqr && n <= m) orgqr_and_hash("geqrf+orgqr", p, A, tau);
        } catch (const std::exception& e) { throw_line("geqrf", tname<T>(), p, e); }
    }
}
// geqrf_blocked_dispatch with a forced panel leaf (internal API), public gemm for the trailing update.
template <class T> void run_geqrf_leaf() {
    using namespace batchlas::sycl_geqrf;
    for (auto [m, n] : shapes("shapes", "128x64")) for (auto ls : split(arg("leaf", "auto,resident,global,reg"))) {
        GeqrfPanelLeaf leaf = ls == "resident" ? GeqrfPanelLeaf::Resident : ls == "global" ? GeqrfPanelLeaf::Global
                            : ls == "reg" ? GeqrfPanelLeaf::Register : GeqrfPanelLeaf::Auto;
        unsigned dbg = geqrf_blocked_debug_params<T>(*g_q, m, n);
        std::string p = P({{"m", I(m)}, {"n", I(n)}, {"leaf", ls}, {"nb", I(dbg & 0xffff)},
                           {"autoleaf", I(dbg >> 16)}, {"regcols", I(geqrf_panel_reg_cols<T>())},
                           {"regmaxm", I(geqrf_panel_reg_max_m<T>())},
                           {"regpref", I(geqrf_panel_reg_preferred<T>(m, std::min(n, (int)(dbg & 0xffff))))},
                           {"batch", I(g_batch)}});
        try {
            Buf<T> A(m, n, g_batch); fill_gen(A, 31 + m * 7 + n, true);
            Flat<T> tau(std::min(m, n), g_batch, poison<T>());
            size_t ws = geqrf_blocked_buffer_size<T>(*g_q, A.view());
            UnifiedVector<std::byte> w(std::max<size_t>(ws, 1));
            geqrf_blocked_dispatch<T>(*g_q, A.view(), tau.span(), Span<std::byte>(w.data(), ws),
                [](Queue& c2, const MV<T>& ga, const MV<T>& gb, const MV<T>& gc, T al, T be, Transpose ta,
                   Transpose tb, ComputePrecision pr) { return gemm<BK, T>(c2, ga, gb, gc, al, be, ta, tb, pr); },
                leaf);
            g_q->wait();
            line("geqrf_leaf", tname<T>(), p, "A", A.hash());
            line("geqrf_leaf", tname<T>(), p, "tau", tau.hash());
        } catch (const std::exception& e) { throw_line("geqrf_leaf", tname<T>(), p, e); }
    }
}
// orgqr on geqrf output (driver pins BATCHLAS_GEQRF_ROUTE=vendor so the input is tree-independent).
template <class T> void run_orgqr() {
    for (auto [m, n] : shapes("shapes", "64x64")) {
        std::string p = P({{"m", I(m)}, {"n", I(n)}, {"batch", I(g_batch)}});
        try {
            Buf<T> A(m, n, g_batch); fill_gen(A, 41 + m * 7 + n, false);
            Flat<T> tau(std::min(m, n), g_batch, poison<T>());
            size_t ws = geqrf_buffer_size<BK, T>(*g_q, A.view(), tau.span());
            UnifiedVector<std::byte> w(std::max<size_t>(ws, 1));
            geqrf<BK, T>(*g_q, A.view(), tau.span(), Span<std::byte>(w.data(), ws));
            g_q->wait();
            line("orgqr", tname<T>(), p, "geqrf_input_A", A.hash());
            orgqr_and_hash("orgqr", p, A, tau);
        } catch (const std::exception& e) { throw_line("orgqr", tname<T>(), p, e); }
    }
}
// ---------------------------------------------------------------- ormqx_cta / ormqr
template <class T> Transpose ctrans() { return is_cx<T> ? Transpose::ConjTrans : Transpose::Trans; }
template <class T> void fill_tau(Flat<T>& tau, uint64_t seed) {
    Rng g(seed);
    for (size_t i = 0; i < (size_t)tau.len * tau.batch; ++i) {
        T x = rnd<T>(g);
        if constexpr (is_cx<T>) tau.v[i] = T(R<T>(1.0 + 0.5 * std::real(x)), R<T>(0.3) * std::imag(x));
        else tau.v[i] = T(1.0 + 0.5 * x);
    }
}
template <class T> void run_ormqx_cta() {
    for (int n : ints("n", "16")) for (auto ss : split(arg("side", "L,R"))) for (auto ts : split(arg("trans", "N,C")))
    for (auto fs : split(arg("fact", "U,L"))) {
        Side side = ss == "L" ? Side::Left : Side::Right;
        Transpose tr = ts == "N" ? Transpose::NoTrans : ctrans<T>();
        Uplo fact = fs == "U" ? Uplo::Upper : Uplo::Lower;
        std::string p = P({{"n", I(n)}, {"side", ss}, {"trans", ts}, {"fact", fs}, {"k", I(n)}, {"batch", I(g_batch)}});
        try {
            Buf<T> A(n, n, g_batch); fill_gen(A, 51 + n, false);
            for (int b = 0; b < g_batch; ++b) for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) A.at(i, j, b) *= R<T>(0.3);
            Flat<T> tau(n, g_batch, poison<T>()); fill_tau(tau, 61 + n);
            Buf<T> C(n, n, g_batch); fill_gen(C, 71 + n, false);
            VectorView<T> tv(tau.v.data(), n, g_batch, 1, n);
            UnifiedVector<std::byte> w(1);
            ormqx_cta<BK, T>(*g_q, A.view(), tv, C.view(), fact, side, tr, n, w.to_span());
            g_q->wait();
            line("ormqx_cta", tname<T>(), p, "C", C.hash());
        } catch (const std::exception& e) { throw_line("ormqx_cta", tname<T>(), p, e); }
    }
}
template <class T> void run_ormqr() {
    // shape nq x k reflectors, C is nq x nc (Left) or nc x nq (Right)
    for (auto [nq, k] : shapes("shapes", "96x64")) for (auto ss : split(arg("side", "L,R"))) for (auto ts : split(arg("trans", "N,C"))) {
        const int nc = std::stoi(arg("nc", "40"));
        Side side = ss == "L" ? Side::Left : Side::Right;
        Transpose tr = ts == "N" ? Transpose::NoTrans : ctrans<T>();
        std::string p = P({{"nq", I(nq)}, {"k", I(k)}, {"nc", I(nc)}, {"side", ss}, {"trans", ts}, {"batch", I(g_batch)}});
        try {
            Buf<T> A(nq, k, g_batch); fill_gen(A, 81 + nq + k, false);
            for (int b = 0; b < g_batch; ++b) for (int j = 0; j < k; ++j) for (int i = 0; i < nq; ++i) A.at(i, j, b) *= R<T>(0.2);
            Flat<T> tau(k, g_batch, poison<T>()); fill_tau(tau, 91 + nq + k);
            Buf<T> C(side == Side::Left ? nq : nc, side == Side::Left ? nc : nq, g_batch); fill_gen(C, 101 + nq, false);
            size_t ws = ormqr_buffer_size<BK, T>(*g_q, A.view(), C.view(), side, tr, tau.span());
            UnifiedVector<std::byte> w(std::max<size_t>(ws, 1));
            ormqr<BK, T>(*g_q, A.view(), C.view(), side, tr, tau.span(), Span<std::byte>(w.data(), ws));
            g_q->wait();
            line("ormqr", tname<T>(), p, "C", C.hash());
        } catch (const std::exception& e) { throw_line("ormqr", tname<T>(), p, e); }
    }
}
// ---------------------------------------------------------------- syev
template <class T> void run_syev() {
    for (int n : ints("n", "16")) for (auto us : split(arg("uplo", "L,U"))) for (auto js : split(arg("jobz", "V,N"))) {
        Uplo uplo = us == "L" ? Uplo::Lower : Uplo::Upper;
        JobType jz = js == "V" ? JobType::EigenVectors : JobType::NoEigenVectors;
        std::string p = P({{"n", I(n)}, {"uplo", us}, {"jobz", js}, {"batch", I(g_batch)}});
        try {
            Buf<T> A(n, n, g_batch); fill_herm(A, 111 + n);
            Flat<R<T>> W(n, g_batch, poison<R<T>>());
            UnifiedVector<int32_t> info(g_batch, -12345);
            size_t ws = syev_buffer_size<BK, T>(*g_q, A.view(), W.span(), jz, uplo);
            UnifiedVector<std::byte> w(std::max<size_t>(ws, 1));
            syev<BK, T>(*g_q, A.view(), W.span(), jz, uplo, Span<std::byte>(w.data(), ws), info.to_span());
            g_q->wait();
            line("syev", tname<T>(), p, "A", A.hash());
            line("syev", tname<T>(), p, "w", W.hash());
            hash_info<T>("syev", p, info);
        } catch (const std::exception& e) { throw_line("syev", tname<T>(), p, e); }
    }
}
// tridiagonal_solver (real only): graded / clustered tridiagonals.
template <class T> void fill_tridiag(UnifiedVector<T>& d, UnifiedVector<T>& e, int n, int batch, int estride, uint64_t seed) {
    for (int b = 0; b < batch; ++b) {
        Rng g(seed * 1000003 + b);
        const int kind = b % 4;
        for (int i = 0; i < n; ++i) {
            double gi = std::pow(1e-4, (double)i / std::max(1, n - 1));
            double dv = kind == 0 ? g.uni() : kind == 1 ? ((i % 3) ? 1.0 : -1.0) * gi : kind == 2 ? 2.0 : (double)(std::abs(n / 2 - i));
            d[(size_t)b * n + i] = T(dv);
            if (i < n - 1) {
                double ev = kind == 0 ? g.uni() : kind == 1 ? 0.5 * gi : kind == 2 ? -1.0 : 1.0;  // kind 3: Wilkinson
                e[(size_t)b * estride + i] = T(ev);
            }
        }
    }
}
template <class T> void run_tridiag() {
    for (int n : ints("n", "32")) for (auto js : split(arg("jobz", "V,N"))) {
        JobType jz = js == "V" ? JobType::EigenVectors : JobType::NoEigenVectors;
        std::string p = P({{"n", I(n)}, {"jobz", js}, {"batch", I(g_batch)}});
        try {
            UnifiedVector<T> d((size_t)n * g_batch + 7, poison<T>()), e((size_t)n * g_batch + 7, poison<T>());
            fill_tridiag(d, e, n, g_batch, n, 121 + n);
            for (int b = 0; b < g_batch; ++b) e[(size_t)b * n + n - 1] = T(0);
            Flat<T> W(n, g_batch, poison<T>());
            Buf<T> Q(n, n, g_batch);
            size_t ws = tridiagonal_solver_buffer_size<BK, T>(*g_q, n, g_batch, jz);
            UnifiedVector<std::byte> w(std::max<size_t>(ws, 1));
            tridiagonal_solver<BK, T>(*g_q, Span<T>(d.data(), (size_t)n * g_batch), Span<T>(e.data(), (size_t)n * g_batch),
                                      W.span(), Span<std::byte>(w.data(), ws), jz, Q.view(), n, g_batch);
            g_q->wait();
            line("tridiag_solver", tname<T>(), p, "w", W.hash());
            line("tridiag_solver", tname<T>(), p, "Q", Q.hash());
            line("tridiag_solver", tname<T>(), p, "d_after", fnv(d.data(), d.size() * sizeof(T)));
            line("tridiag_solver", tname<T>(), p, "e_after", fnv(e.data(), e.size() * sizeof(T)));
        } catch (const std::exception& e) { throw_line("tridiag_solver", tname<T>(), p, e); }
    }
}
// stebz (All / Index / Value) then stein on its output (real only).
template <class T> void run_stebz_stein() {
    for (int n : ints("n", "32")) for (auto rs : split(arg("range", "all,index,value"))) {
        StebzParams<T> bp;
        int cap = n;
        if (rs == "index") { bp.range = EigenRangeType::Index; bp.il = n / 4; bp.iu = n / 2; cap = n; }
        if (rs == "value") { bp.range = EigenRangeType::Value; bp.vl = T(-0.5); bp.vu = T(1.25); cap = n; }
        std::string p = P({{"n", I(n)}, {"range", rs}, {"batch", I(g_batch)}});
        try {
            UnifiedVector<T> d((size_t)n * g_batch + 7, poison<T>()), e((size_t)(n - 1) * g_batch + 7, poison<T>());
            fill_tridiag(d, e, n, g_batch, n - 1, 131 + n);
            VectorView<T> dv(d.data(), n, g_batch, 1, n), ev(e.data(), n - 1, g_batch, 1, n - 1);
            Flat<T> W(cap, g_batch, poison<T>());
            VectorView<T> wv(W.v.data(), cap, g_batch, 1, cap);
            UnifiedVector<int32_t> m(g_batch, -12345);
            size_t ws = stebz_buffer_size<BK, T>(*g_q, n, g_batch, bp);
            UnifiedVector<std::byte> w(std::max<size_t>(ws, 1));
            stebz<BK, T>(*g_q, dv, ev, wv, m.to_span(), Span<std::byte>(w.data(), ws), bp);
            g_q->wait();
            line("stebz", tname<T>(), p, "w", W.hash());
            line("stebz", tname<T>(), p, "m", fnv(m.data(), m.size() * 4));
            // stein with per-item counts from stebz
            SteinParams<T> sp;
            Buf<T> Z(n, cap, g_batch);
            size_t ws2 = stein_buffer_size<BK, T>(*g_q, n, cap, g_batch, sp);
            UnifiedVector<std::byte> w2(std::max<size_t>(ws2, 1));
            if (rs == "all")
                stein<BK, T>(*g_q, dv, ev, wv, cap, Z.view(), Span<std::byte>(w2.data(), ws2), sp);
            else
                stein<BK, T>(*g_q, dv, ev, wv, cap, Span<const int32_t>(m.data(), m.size()), Z.view(),
                             Span<std::byte>(w2.data(), ws2), sp);
            g_q->wait();
            line("stein", tname<T>(), p, "Z", Z.hash());
        } catch (const std::exception& e) { throw_line("stebz_stein", tname<T>(), p, e); }
    }
}
template <class T> void run_syevx() {
    for (int n : ints("n", "32")) for (auto rs : split(arg("range", "index,value"))) for (auto js : split(arg("jobz", "V,N"))) {
        JobType jz = js == "V" ? JobType::EigenVectors : JobType::NoEigenVectors;
        SyevxParams<T> sp;
        size_t cap = (size_t)n;
        if (rs == "index") { sp.select = SyevxSelect::Index; sp.il = n / 4; sp.iu = n / 2; cap = n / 2 - n / 4 + 1; }
        else { sp.select = SyevxSelect::Value; sp.vl = R<T>(-1.0); sp.vu = R<T>(2.0); cap = n; }
        std::string p = P({{"n", I(n)}, {"range", rs}, {"jobz", js}, {"cap", I((long)cap)}, {"batch", I(g_batch)}});
        try {
            Buf<T> A(n, n, g_batch); fill_herm(A, 141 + n);
            for (int b = 0; b < g_batch; ++b) for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i)  // mirror: syevx reads full A?
                if (i < j) A.at(i, j, b) = conj_(A.at(j, i, b));
            Flat<R<T>> W((int)cap, g_batch, poison<R<T>>());
            UnifiedVector<int32_t> m(g_batch, -12345), info(g_batch, -12345);
            Buf<T> V(n, (int)cap, g_batch);
            MV<T> Vv = jz == JobType::EigenVectors ? V.view() : MV<T>();
            size_t ws = syevx_direct_subset_buffer_size<BK, T, MatrixFormat::Dense>(*g_q, A.view(), W.span(), cap, jz, Vv, sp);
            UnifiedVector<std::byte> w(std::max<size_t>(ws, 1));
            syevx_direct_subset<BK, T, MatrixFormat::Dense>(*g_q, A.view(), W.span(), m.to_span(), cap,
                                                            Span<std::byte>(w.data(), ws), jz, Vv, sp, info.to_span());
            g_q->wait();
            line("syevx_subset", tname<T>(), p, "w", W.hash());
            line("syevx_subset", tname<T>(), p, "m", fnv(m.data(), m.size() * 4));
            if (jz == JobType::EigenVectors) line("syevx_subset", tname<T>(), p, "V", V.hash());
            hash_info<T>("syevx_subset", p, info);
        } catch (const std::exception& e) { throw_line("syevx_subset", tname<T>(), p, e); }
    }
}
// ---------------------------------------------------------------- gesvd
template <class T> void run_gesvd() {
    const bool herm = arg("herm", "0") == "1";
    for (auto [m, n] : shapes("shapes", "32x32")) for (auto jb : split(arg("jobs", "A,S,N"))) for (auto us : split(arg("uplo", herm ? "L,U" : "-"))) {
        if (herm && m != n) continue;
        SvdVectors job = jb == "A" ? SvdVectors::All : jb == "S" ? SvdVectors::Thin : SvdVectors::None;
        const int k = std::min(m, n);
        std::string p = P({{"m", I(m)}, {"n", I(n)}, {"job", jb}, {"herm", herm ? us : std::string("0")}, {"batch", I(g_batch)}});
        try {
            Buf<T> A(m, n, g_batch);
            if (herm) fill_herm(A, 151 + n); else fill_gen(A, 161 + m * 3 + n, true);
            Flat<R<T>> S(k, g_batch, poison<R<T>>());
            Buf<T> Ub(m, job == SvdVectors::All ? m : k, g_batch), Vb(job == SvdVectors::All ? n : k, n, g_batch);
            MV<T> Uv = job == SvdVectors::None ? MV<T>() : Ub.view();
            MV<T> Vv = job == SvdVectors::None ? MV<T>() : Vb.view();
            UnifiedVector<int32_t> info(g_batch, -12345);
            size_t ws;
            Uplo hu = us == "U" ? Uplo::Upper : Uplo::Lower;
            if (herm) ws = gesvd_buffer_size<BK, T>(*g_q, A.view(), S.span(), Uv, Vv, job, job, hu);
            else ws = gesvd_buffer_size<BK, T>(*g_q, A.view(), S.span(), Uv, Vv, job, job);
            UnifiedVector<std::byte> w(std::max<size_t>(ws, 1));
            if (herm) gesvd<BK, T>(*g_q, A.view(), S.span(), Uv, Vv, job, job, hu, Span<std::byte>(w.data(), ws), info.to_span());
            else gesvd<BK, T>(*g_q, A.view(), S.span(), Uv, Vv, job, job, Span<std::byte>(w.data(), ws), info.to_span());
            g_q->wait();
            line("gesvd", tname<T>(), p, "S", S.hash());
            if (job != SvdVectors::None) { line("gesvd", tname<T>(), p, "U", Ub.hash()); line("gesvd", tname<T>(), p, "Vh", Vb.hash()); }
            line("gesvd", tname<T>(), p, "A_after", A.hash());
            hash_info<T>("gesvd", p, info);
        } catch (const std::exception& e) { throw_line("gesvd", tname<T>(), p, e); }
    }
}
template <class T> void run_gebrd() {
    for (auto [m, n] : shapes("shapes", "32x32")) for (auto vs : split(arg("variant", "unblocked,cta,blocked"))) {
        const int k = std::min(m, n);
        std::string p = P({{"m", I(m)}, {"n", I(n)}, {"variant", vs}, {"batch", I(g_batch)}});
        try {
            Buf<T> A(m, n, g_batch); fill_gen(A, 171 + m * 3 + n, true);
            Flat<R<T>> d(k, g_batch, poison<R<T>>()), e(k, g_batch, poison<R<T>>());
            Flat<T> tq(k, g_batch, poison<T>()), tp(k, g_batch, poison<T>());
            VectorView<R<T>> dv(d.v.data(), k, g_batch, 1, k), ev(e.v.data(), std::max(k - 1, 1), g_batch, 1, k);
            VectorView<T> qv(tq.v.data(), k, g_batch, 1, k), pv(tp.v.data(), k, g_batch, 1, k);
            if (vs == "unblocked") gebrd_unblocked<BK, T>(*g_q, A.view(), dv, ev, qv, pv);
            else if (vs == "cta") gebrd_cta<BK, T>(*g_q, A.view(), dv, ev, qv, pv);
            else {
                const int bs = std::stoi(arg("bs", "16"));
                size_t ws = gebrd_blocked_buffer_size<BK, T>(*g_q, A.view(), dv, ev, qv, pv, bs);
                UnifiedVector<std::byte> w(std::max<size_t>(ws, 1));
                gebrd_blocked<BK, T>(*g_q, A.view(), dv, ev, qv, pv, Span<std::byte>(w.data(), ws), bs);
                g_q->wait();
            }
            g_q->wait();
            line("gebrd", tname<T>(), p, "A", A.hash());
            line("gebrd", tname<T>(), p, "d", d.hash());
            line("gebrd", tname<T>(), p, "e", e.hash());
            line("gebrd", tname<T>(), p, "tauq", tq.hash());
            line("gebrd", tname<T>(), p, "taup", tp.hash());
        } catch (const std::exception& e) { throw_line("gebrd", tname<T>(), p, e); }
    }
}

#define BITID_EACH2(fn)                                \
    do {                                               \
        if (want_type("f")) fn<float>();               \
        if (want_type("d")) fn<double>();              \
    } while (0)
#define BITID_EACH(fn)                                         \
    do {                                                       \
        BITID_EACH2(fn);                                       \
        if (want_type("cf")) fn<std::complex<float>>();        \
        if (want_type("cd")) fn<std::complex<double>>();       \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: bitid <group> [key=val ...]\n"); return 2; }
    std::string group = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i]; auto eq = a.find('=');
        if (eq != std::string::npos) g_args[a.substr(0, eq)] = a.substr(eq + 1);
    }
    g_tag = arg("tag", "-");
    g_batch = std::stoi(arg("batch", "1024"));
    Queue q(Device("gpu"), BK, true);
    g_q = &q;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    {
        if (group == "potrf") BITID_EACH(run_potrf);
        else if (group == "potri") BITID_EACH(run_potri);
        else if (group == "geqrf") BITID_EACH(run_geqrf);
        else if (group == "geqrf_leaf") BITID_EACH(run_geqrf_leaf);
        else if (group == "orgqr") BITID_EACH(run_orgqr);
        else if (group == "ormqx_cta") BITID_EACH(run_ormqx_cta);
        else if (group == "ormqr") BITID_EACH(run_ormqr);
        else if (group == "syev") BITID_EACH(run_syev);
        else if (group == "syevx") BITID_EACH(run_syevx);
        else if (group == "gesvd") BITID_EACH(run_gesvd);
        else if (group == "gebrd") BITID_EACH(run_gebrd);
        else if (group == "tridiag") BITID_EACH2(run_tridiag);  // real-only entry points
        else if (group == "stebz_stein") BITID_EACH2(run_stebz_stein);
        else { std::fprintf(stderr, "unknown group %s\n", group.c_str()); return 2; }
    }
    q.wait();
    return 0;
}
