// WP6 BASELINE: time the PUBLIC getrf / getrs / getri against cuBLAS, and time
// a routed-trsm composition of getrs / getri on the same shapes.
//
// Modelled on experiments/wp5_qr/bench/qrbench.cpp (itself
// experiments/wp4_potrf/phase2_ab/realpotrf.cpp's pattern): public entry points
// only, correctness checked in the SAME PROCESS so a fast wrong answer cannot be
// reported as a win, NaN-propagating probes, and every route printed on the row
// so a pin that did not take is visible.
//
// The oracle is a HOST reference on the factorization identity, never
// vendor-vs-vendor:
//   getrf : || (P A0) x - L (U x) ||inf / || A0 x ||inf, P rebuilt on the host
//           from the pivot list the device wrote. Independent of who factored.
//   getrs : || A0 X[:,j] - B0[:,j] ||inf / || B0[:,j] ||inf, A0 the ORIGINAL.
//   getri : || A0 (C e) - e ||inf / ||e||inf.
// All three read A0, which no device call ever writes.
//
// THE PIVOT FORMAT IS BACKEND-DEPENDENT. cublas.cc:1509 does
// pivots.as_span<int>() -- cuBLAS writes int32 PACKED AT THE FRONT of the
// caller's int64 buffer. The host probes therefore read the buffer as int32.
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getri.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/sycl_interop.hh>

#include <batchlas/blas/dispatch/vendor_available.hh>
#include "src/backends/trsm_route.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace batchlas;
static constexpr Backend BE = Backend::CUDA;
static constexpr bool kVendorL3 = dispatch::level3_vendor_available<BE>;

template <class T> struct Prom { using type = double; };
template <class R> struct Prom<std::complex<R>> { using type = std::complex<double>; };

static inline double ab(double x) { return std::fabs(x); }
static inline double ab(std::complex<double> x) { return std::abs(x); }
static inline double up(float x) { return double(x); }
static inline double up(double x) { return x; }
static inline std::complex<double> up(std::complex<float> x) { return {double(x.real()), double(x.imag())}; }
static inline std::complex<double> up(std::complex<double> x) { return x; }

// NAN-PROPAGATING max. std::max(a,b) returns a when b is NaN, so a poisoned
// probe reads as a perfect one (WP5 break K5).
static inline double nanmax(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::quiet_NaN();
    return a > b ? a : b;
}

template <class T> static inline T mk(double re, double im);
template <> inline float mk<float>(double re, double) { return float(re); }
template <> inline double mk<double>(double re, double) { return re; }
template <> inline std::complex<float> mk<std::complex<float>>(double re, double im) { return {float(re), float(im)}; }
template <> inline std::complex<double> mk<std::complex<double>>(double re, double im) { return {re, im}; }

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(int32_t(uint32_t(s >> 32))) / 2147483648.0;
    }
};

struct Stat { double med, mean, relsd, min; };
static Stat stat_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const double med = v[v.size() / 2];
    double mean = 0; for (double x : v) mean += x; mean /= double(v.size());
    double sd = 0; for (double x : v) sd += (x - mean) * (x - mean);
    sd = std::sqrt(sd / double(v.size()));
    return {med, mean, mean > 0 ? sd / mean : 0.0, v.front()};
}

static double warm_s() { const char* e = std::getenv("WARM_S"); return e ? std::atof(e) : 1.5; }
static int nprobe()    { const char* e = std::getenv("NPROBE"); return e ? std::atoi(e) : 2; }

// DELIBERATE BREAKS, so the oracle is not a blind guard. Each one corrupts the
// exact thing one probe claims to check; a break that leaves a residual small is
// a finding, not a pass. Never set in a measurement run -- the flag is printed
// nowhere, so run_break.sh keeps its output in its own file.
//   piv    : the getrf probe ignores the pivot list (P := I).
//   factor : zero the strict LOWER triangle of the factor before probing.
//   sol    : perturb the solution by 1e-2 relative before probing.
//   laswp  : the composition skips its row interchange.
static bool brk(const char* w) {
    const char* e = std::getenv("BREAK");
    return e && std::strcmp(e, w) == 0;
}

static const char* orig_name(dispatch::Origin o) {
    switch (o) { case dispatch::Origin::Auto: return "auto";
                 case dispatch::Origin::Native: return "native";
                 case dispatch::Origin::Vendor: return "vendor"; }
    return "?";
}
static const char* alg_name(dispatch::Algorithm a) {
    switch (a) { case dispatch::Algorithm::Auto: return "auto";
                 case dispatch::Algorithm::Direct: return "direct";
                 case dispatch::Algorithm::CTA: return "cta";
                 case dispatch::Algorithm::Blocked: return "blocked";
                 default: return "other"; }
}
static std::string rstr(dispatch::Route r) {
    return std::string(orig_name(r.origin)) + ":" + alg_name(r.algo);
}

// ---------------------------------------------------------------- laswp
// The row-interchange kernel a native getrs would need. LAPACK ipiv is a
// SEQUENCE OF INTERCHANGES, not a permutation, so it must be applied in order;
// one work-item per (batch item, column) walks k = 0..n-1 serially.
template <typename T>
static void laswp_forward(Queue& ctx, T* B, int ldb, int64_t strideB, int n, int nrhs,
                          const int* piv, int64_t piv_stride, int batch) {
    sycl::queue& q = sycl_queue(ctx);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(size_t(batch), size_t(nrhs)), [=](sycl::item<2> it) {
            const size_t b = it[0], j = it[1];
            T* col = B + b * size_t(strideB) + j * size_t(ldb);
            const int* p = piv + b * size_t(piv_stride);
            for (int k = 0; k < n; ++k) {
                const int ip = p[k] - 1;   // 1-BASED
                if (ip != k) { T t = col[k]; col[k] = col[ip]; col[ip] = t; }
            }
        });
    });
}

// LASWP=gather. The interchange LIST collapsed to a PERMUTATION once, then one
// coalesced pass instead of ~n scattered swaps.
//
// This exists because BREAK=laswp accidentally measured the interchange kernel
// above and found it is HALF of a composed getrs at n=128 (0.446 ms -> 0.225 ms
// with the swap removed). The reason is structural, not a bug in the kernel: in
// column-major, one work-item per column walks its own column contiguously but
// consecutive work-items are `ldb` apart, so every warp access is 32 separate
// transactions. The list is inherently sequential in k, so it cannot be
// parallelised -- but it can be COLLAPSED. Applying the interchanges to an
// identity index array gives a permutation; a gather under that permutation has
// consecutive work-items on consecutive ROWS, which in column-major is
// contiguous, so the write is fully coalesced and the read is a gather inside
// one column (n elements, cache-resident).
//
// Cost: one int32[n] per batch item, plus -- for getrs only -- an out-of-place
// RHS. getri needs neither, because the permuted identity can be WRITTEN
// directly and never permuted at all.
static void perm_build(Queue& ctx, int* perm, const int* piv, int64_t piv_stride,
                       int n, int batch) {
    sycl::queue& q = sycl_queue(ctx);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(size_t(batch)), [=](sycl::item<1> it) {
            const size_t b = it[0];
            int* p = perm + b * size_t(n);
            const int* pv = piv + b * size_t(piv_stride);
            for (int i = 0; i < n; ++i) p[i] = i;
            for (int k = 0; k < n; ++k) {
                const int ip = pv[k] - 1;
                if (ip != k) { const int t = p[k]; p[k] = p[ip]; p[ip] = t; }
            }
        });
    });
}

template <typename T>
static void gather_rows(Queue& ctx, const T* src, T* dst, int ld, int64_t stride,
                        int n, int ncols, const int* perm, int batch) {
    sycl::queue& q = sycl_queue(ctx);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(size_t(batch), size_t(n) * size_t(ncols)),
                       [=](sycl::item<2> it) {
            const size_t b = it[0], e = it[1];
            const size_t i = e % size_t(n), j = e / size_t(n);
            dst[b * size_t(stride) + j * size_t(ld) + i] =
                src[b * size_t(stride) + j * size_t(ld) + size_t(perm[b * size_t(n) + i])];
        });
    });
}

// P written straight into C: getri's permutation costs one store per element,
// the same store fill_identity already had to do.
template <typename T>
static void fill_permuted_identity(Queue& ctx, T* C, int ldc, int64_t strideC,
                                   int n, const int* perm, int batch) {
    sycl::queue& q = sycl_queue(ctx);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(size_t(batch), size_t(n) * size_t(n)), [=](sycl::item<2> it) {
            const size_t b = it[0], e = it[1];
            const size_t i = e % size_t(n), j = e / size_t(n);
            C[b * size_t(strideC) + j * size_t(ldc) + i] =
                (size_t(perm[b * size_t(n) + i]) == j) ? T(1) : T(0);
        });
    });
}

// Fill C with the identity, batched.
template <typename T>
static void fill_identity(Queue& ctx, T* C, int ldc, int64_t strideC, int n, int batch) {
    sycl::queue& q = sycl_queue(ctx);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(size_t(batch), size_t(n) * size_t(n)), [=](sycl::item<2> it) {
            const size_t b = it[0], e = it[1];
            const size_t i = e % size_t(n), j = e / size_t(n);
            C[b * size_t(strideC) + j * size_t(ldc) + i] = (i == j) ? T(1) : T(0);
        });
    });
}

// ANTI-VACUITY on the CONFIGURATION: how many of item 0's n pivots are not the
// diagonal. Zero means partial pivoting never moved a row, so every probe below
// is blind to the pivot path no matter how carefully it is written. Necessary,
// and (as the WP4/WP5 record insists) not sufficient -- which is why the BREAK
// runs exist as well.
static int nontrivial_pivots(const int* piv, int64_t piv_stride, int n) {
    int c = 0;
    for (int k = 0; k < n; ++k) if (piv[k] != k + 1) ++c;
    (void)piv_stride;
    return c;
}

// ---------------------------------------------------------------- probes
// || (P A0) x - L (U x) ||inf / || A0 x ||inf, P rebuilt from the DEVICE pivots.
template <typename T>
static double getrf_probe(const UnifiedVector<T>& F, const UnifiedVector<T>& A0,
                          const int* piv, int64_t piv_stride,
                          int n, int batch, int np) {
    using D = typename Prom<T>::type;
    const size_t st = size_t(n) * n;
    double worst = 0;
    for (int b : {0, batch - 1}) {
        for (int p = 0; p < np; ++p) {
            Rng rg(uint64_t(b) * 977 + uint64_t(p) * 31 + 7);
            std::vector<D> x(n);
            for (int j = 0; j < n; ++j) x[j] = up(mk<T>(rg.next(), rg.next()));
            std::vector<D> ref(n, D(0));
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    ref[i] += up(A0[size_t(b) * st + size_t(j) * n + i]) * x[j];
            // (P A0) x : apply the interchanges to A0 x, in order.
            const int* pv = piv + size_t(b) * size_t(piv_stride);
            for (int k = 0; brk("piv") ? false : k < n; ++k) {
                const int ip = pv[k] - 1;
                if (ip < 0 || ip >= n) return std::numeric_limits<double>::quiet_NaN();
                if (ip != k) std::swap(ref[k], ref[ip]);
            }
            // y = U x ; z = L y
            std::vector<D> y(n, D(0)), z(n, D(0));
            for (int i = 0; i < n; ++i) {
                D acc = D(0);
                for (int j = i; j < n; ++j) acc += up(F[size_t(b) * st + size_t(j) * n + i]) * x[j];
                y[i] = acc;
            }
            for (int i = 0; i < n; ++i) {
                D acc = y[i];                       // unit diagonal
                for (int j = 0; j < i; ++j) acc += up(F[size_t(b) * st + size_t(j) * n + i]) * y[j];
                z[i] = acc;
            }
            double num = 0, den = 0;
            for (int i = 0; i < n; ++i) { num = nanmax(num, ab(z[i] - ref[i])); den = nanmax(den, ab(ref[i])); }
            worst = nanmax(worst, den > 0 ? num / den : num);
        }
    }
    return worst;
}

// || A0 X[:,j] - B0[:,j] ||inf / || B0[:,j] ||inf on a few RHS columns.
template <typename T>
static double solve_probe(const UnifiedVector<T>& X, const UnifiedVector<T>& B0,
                          const UnifiedVector<T>& A0, int n, int nrhs, int batch, int np) {
    using D = typename Prom<T>::type;
    const size_t sa = size_t(n) * n, sb = size_t(n) * nrhs;
    double worst = 0;
    for (int b : {0, batch - 1}) {
        for (int p = 0; p < np; ++p) {
            const int j = (p * 7) % nrhs;
            std::vector<D> r(n, D(0));
            for (int c = 0; c < n; ++c)
                for (int i = 0; i < n; ++i)
                    r[i] += up(A0[size_t(b) * sa + size_t(c) * n + i]) * up(X[size_t(b) * sb + size_t(j) * n + c]);
            double num = 0, den = 0;
            for (int i = 0; i < n; ++i) {
                const D bi = up(B0[size_t(b) * sb + size_t(j) * n + i]);
                num = nanmax(num, ab(r[i] - bi)); den = nanmax(den, ab(bi));
            }
            worst = nanmax(worst, den > 0 ? num / den : num);
        }
    }
    return worst;
}

// || A0 (C e) - e ||inf / ||e||inf.
template <typename T>
static double inv_probe(const UnifiedVector<T>& C, const UnifiedVector<T>& A0,
                        int n, int batch, int np) {
    using D = typename Prom<T>::type;
    const size_t st = size_t(n) * n;
    double worst = 0;
    for (int b : {0, batch - 1}) {
        for (int p = 0; p < np; ++p) {
            Rng rg(uint64_t(b) * 613 + uint64_t(p) * 17 + 3);
            std::vector<D> e(n);
            for (int j = 0; j < n; ++j) e[j] = up(mk<T>(rg.next(), rg.next()));
            std::vector<D> z(n, D(0)), w(n, D(0));
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) z[i] += up(C[size_t(b) * st + size_t(j) * n + i]) * e[j];
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) w[i] += up(A0[size_t(b) * st + size_t(j) * n + i]) * z[j];
            double num = 0, den = 0;
            for (int i = 0; i < n; ++i) { num = nanmax(num, ab(w[i] - e[i])); den = nanmax(den, ab(e[i])); }
            worst = nanmax(worst, den > 0 ? num / den : num);
        }
    }
    return worst;
}

// ---------------------------------------------------------------- driver
template <typename T>
static int run(const std::string& mode, const char* tn, int n, int nrhs, int batch, int reps) {
    auto q = std::make_shared<Queue>(Device("gpu"), BE);
    const size_t sa = size_t(n) * n;
    const size_t sb = size_t(n) * nrhs;

    UnifiedVector<T> A0(sa * batch), A(sa * batch);
    UnifiedVector<int64_t> piv(size_t(n) * batch);
    UnifiedVector<int32_t> info(batch);
    {
        // DIAGONALLY DOMINANT, THEN ROW-PERMUTED. The dominance makes the
        // factorization well conditioned so the residual measures the KERNEL and
        // not the matrix; the permutation is what makes the residual able to SEE
        // the pivoting at all.
        //
        // This is a recorded break, not a precaution. The first version of this
        // harness used the diagonally dominant matrix alone, and on it partial
        // pivoting selects the diagonal at every step, so ipiv is the identity:
        // BREAK=piv (probe ignores the pivot list) and BREAK=laswp (composition
        // drops its row interchange) both left the residual bit-identical at
        // 2.446e-07 / 1.055e-15. A pivot path that is never exercised is exactly
        // the blind guard this repository keeps shipping. Permuting the rows by a
        // per-item random permutation leaves the conditioning untouched and forces
        // partial pivoting to undo it, so ipiv is non-trivial by construction --
        // and the anti-vacuity assertion below checks that it actually is.
        Rng rg(12345);
        for (size_t i = 0; i < A0.size(); ++i) A0[i] = mk<T>(rg.next(), rg.next());
        std::vector<T> col(n);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i)
                A0[size_t(b) * sa + size_t(i) * n + i] =
                    A0[size_t(b) * sa + size_t(i) * n + i] + mk<T>(double(n), 0.0);
            // Fisher-Yates on the row index, applied to every column.
            std::vector<int> perm(n);
            for (int i = 0; i < n; ++i) perm[i] = i;
            for (int i = n - 1; i > 0; --i) {
                const int j = int((rg.next() * 0.5 + 0.5) * double(i + 1)) % (i + 1);
                std::swap(perm[i], perm[j]);
            }
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) col[i] = A0[size_t(b) * sa + size_t(j) * n + size_t(perm[i])];
                for (int i = 0; i < n; ++i) A0[size_t(b) * sa + size_t(j) * n + i] = col[i];
            }
        }
    }
    UnifiedVector<T*> pA0(batch), pA(batch);
    MatrixView<T, MatrixFormat::Dense> A0v(A0.data(), n, n, n, int(sa), batch, pA0.data());
    MatrixView<T, MatrixFormat::Dense> Av(A.data(), n, n, n, int(sa), batch, pA.data());
    auto reset_A = [&] { MatrixView<T, MatrixFormat::Dense>::copy(*q, Av, A0v); q->wait(); };

    const size_t f_ws = getrf_buffer_size<BE, T>(*q, Av);
    UnifiedVector<std::byte> fws(f_ws ? f_ws : 1);
    const char* le = std::getenv("LASWP");
    const bool gather = le && std::strcmp(le, "gather") == 0;
    const int* pivi = reinterpret_cast<const int*>(piv.data());
    const int64_t piv_stride_i32 = int64_t(n);   // cuBLAS writes n*batch int32 CONTIGUOUSLY at the front of the int64 buffer, so the per-item int32 stride is n, not 2n

    const double dn = double(n);
    const double gf_getrf = double(batch) * (2.0 / 3.0) * dn * dn * dn;

    if (mode == "getrf") {
        const auto w0 = std::chrono::steady_clock::now();
        do { reset_A(); getrf<BE, T>(*q, Av, piv.to_span(), fws.to_span(), info.to_span()); q->wait(); }
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s());
        std::vector<double> ms;
        for (int r = 0; r < reps; ++r) {
            reset_A();
            const auto t0 = std::chrono::steady_clock::now();
            getrf<BE, T>(*q, Av, piv.to_span(), fws.to_span(), info.to_span());
            q->wait();
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        const Stat s = stat_of(ms);
        int bad = 0; for (int b = 0; b < batch; ++b) if (info[b] != 0) ++bad;
        if (brk("factor"))
            for (int b : {0, batch - 1})
                for (int j = 0; j < n; ++j)
                    for (int i = j + 1; i < n; ++i) A[size_t(b) * sa + size_t(j) * n + i] = T(0);
        const double res = getrf_probe<T>(A, A0, pivi, piv_stride_i32, n, batch, nprobe());
        const int ntp = nontrivial_pivots(pivi, piv_stride_i32, n);
        std::printf("getrf,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.2f,%.3e,%zu,%s,%d,%d,%s\n",
                    tn, n, nrhs, batch, s.med, s.mean, s.relsd, gf_getrf / (s.med * 1e6),
                    res, f_ws, "vendor:auto", bad, ntp,
                    std::isfinite(res) && bad == 0 && ntp > 0 ? "ok" : "BAD");
        return 0;
    }

    // Every remaining mode needs a FACTORED A. Produce it once, UNTIMED, with
    // the vendor getrf -- which is what makes this measurable before any native
    // getrf exists.
    reset_A();
    getrf<BE, T>(*q, Av, piv.to_span(), fws.to_span(), info.to_span());
    q->wait();
    UnifiedVector<T> F(sa * batch);
    std::memcpy(F.data(), A.data(), F.size() * sizeof(T));
    const double fres = getrf_probe<T>(F, A0, pivi, piv_stride_i32, n, batch, nprobe());
    UnifiedVector<T*> pF(batch);
    MatrixView<T, MatrixFormat::Dense> Fv(F.data(), n, n, n, int(sa), batch, pF.data());

    if (mode == "getri" || mode == "getri_trsm") {
        UnifiedVector<T> C(sa * batch);
        UnifiedVector<T*> pC(batch);
        MatrixView<T, MatrixFormat::Dense> Cv(C.data(), n, n, n, int(sa), batch, pC.data());
        const size_t i_ws = getri_buffer_size<BE, T>(*q, Fv);
        UnifiedVector<std::byte> iws(i_ws ? i_ws : 1);

        // The two trsms the composition issues, with their resolved routes.
        const std::string rL = rstr(backend::trsm_route<T>(*q, Fv, Cv, Side::Left, Uplo::Lower,
                                                           Transpose::NoTrans, Diag::Unit, kVendorL3));
        const std::string rU = rstr(backend::trsm_route<T>(*q, Fv, Cv, Side::Left, Uplo::Upper,
                                                           Transpose::NoTrans, Diag::NonUnit, kVendorL3));
        const std::string rt = (mode == "getri") ? std::string("vendor:auto") : (rL + "|" + rU);

        auto do_vendor = [&] { getri<BE, T>(*q, Fv, Cv, piv.to_span(), iws.to_span(), info.to_span()); };
        // getri as getrs against the identity, IN PLACE in C: no extra workspace
        // at all beyond C, which the caller already owns.
        //
        // LASWP=gather writes P straight into C instead of writing I and then
        // permuting it -- same number of stores, one kernel instead of two, and
        // the interchange list never touches the matrix.
        UnifiedVector<int> permbuf(gather ? size_t(n) * batch : 1);
        auto do_comp = [&] {
            if (gather) {
                perm_build(*q, permbuf.data(), pivi, piv_stride_i32, n, batch);
                fill_permuted_identity<T>(*q, C.data(), n, int64_t(sa), n, permbuf.data(), batch);
            } else {
                fill_identity<T>(*q, C.data(), n, int64_t(sa), n, batch);
                if (!brk("laswp")) laswp_forward<T>(*q, C.data(), n, int64_t(sa), n, n, pivi, piv_stride_i32, batch);
            }
            trsm<BE, T>(*q, Fv, Cv, T(1), Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::Unit);
            trsm<BE, T>(*q, Fv, Cv, T(1), Side::Left, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit);
        };
        auto go = [&] { if (mode == "getri") do_vendor(); else do_comp(); };

        const auto w0 = std::chrono::steady_clock::now();
        do { go(); q->wait(); }
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s());
        std::vector<double> ms;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            go();
            q->wait();
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        const Stat s = stat_of(ms);
        if (brk("sol"))
            for (int b : {0, batch - 1})
                for (size_t i = 0; i < sa; ++i)
                    C[size_t(b) * sa + i] = C[size_t(b) * sa + i] * T(1.01);
        const double res = inv_probe<T>(C, A0, n, batch, nprobe());
        const double gf = double(batch) * (4.0 / 3.0) * dn * dn * dn;  // getri ~ 4n^3/3
        const size_t ws = (mode == "getri") ? i_ws
                                            : (gather ? sizeof(int) * size_t(n) * batch : 0);
        const int ntp = nontrivial_pivots(pivi, piv_stride_i32, n);
        std::printf("%s,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.2f,%.3e,%zu,%s,%.3e,%d,%s\n",
                    mode.c_str(), tn, n, nrhs, batch, s.med, s.mean, s.relsd,
                    gf / (s.med * 1e6), res, ws, rt.c_str(), fres, ntp,
                    std::isfinite(res) && std::isfinite(fres) && ntp > 0 ? "ok" : "BAD");
        return 0;
    }

    if (mode == "getrs" || mode == "getrs_trsm") {
        UnifiedVector<T> B0(sb * batch), X(sb * batch);
        { Rng rg(777); for (size_t i = 0; i < B0.size(); ++i) B0[i] = mk<T>(rg.next(), rg.next()); }
        UnifiedVector<T*> pB0(batch), pX(batch);
        MatrixView<T, MatrixFormat::Dense> B0v(B0.data(), n, nrhs, n, int(sb), batch, pB0.data());
        MatrixView<T, MatrixFormat::Dense> Xv(X.data(), n, nrhs, n, int(sb), batch, pX.data());
        const size_t s_ws = getrs_buffer_size<BE, T>(*q, Fv, Xv, Transpose::NoTrans);
        UnifiedVector<std::byte> sws(s_ws ? s_ws : 1);

        const std::string rL = rstr(backend::trsm_route<T>(*q, Fv, Xv, Side::Left, Uplo::Lower,
                                                           Transpose::NoTrans, Diag::Unit, kVendorL3));
        const std::string rU = rstr(backend::trsm_route<T>(*q, Fv, Xv, Side::Left, Uplo::Upper,
                                                           Transpose::NoTrans, Diag::NonUnit, kVendorL3));
        const std::string rt = (mode == "getrs") ? std::string("vendor:auto") : (rL + "|" + rU);

        // LASWP=gather solves out of place: X is gathered into S under the
        // collapsed permutation and the two trsms then run on S. That is the
        // extra workspace the gather costs for getrs -- n*nrhs*batch scalars
        // plus n*batch int32 -- and it is why the getri arm above, which needs
        // neither, is the better bargain.
        UnifiedVector<T> S(gather ? sb * batch : 1);
        UnifiedVector<T*> pS(gather ? batch : 1);
        MatrixView<T, MatrixFormat::Dense> Sv(S.data(), n, nrhs, n, int(sb), batch, pS.data());
        UnifiedVector<int> permbuf(gather ? size_t(n) * batch : 1);
        auto& Rv = gather ? Sv : Xv;
        auto& R = gather ? S : X;

        auto reset_X = [&] { MatrixView<T, MatrixFormat::Dense>::copy(*q, Xv, B0v); };
        auto do_vendor = [&] { getrs<BE, T>(*q, Fv, Xv, Transpose::NoTrans, piv.to_span(), sws.to_span()); };
        auto do_comp = [&] {
            if (gather) {
                perm_build(*q, permbuf.data(), pivi, piv_stride_i32, n, batch);
                gather_rows<T>(*q, X.data(), S.data(), n, int64_t(sb), n, nrhs, permbuf.data(), batch);
            } else if (!brk("laswp")) {
                laswp_forward<T>(*q, X.data(), n, int64_t(sb), n, nrhs, pivi, piv_stride_i32, batch);
            }
            trsm<BE, T>(*q, Fv, Rv, T(1), Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::Unit);
            trsm<BE, T>(*q, Fv, Rv, T(1), Side::Left, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit);
        };
        // The RHS copy is setup, not solve, so it is outside the timed region for
        // BOTH arms -- it would otherwise be charged to whichever arm is faster.
        auto go = [&] { if (mode == "getrs") do_vendor(); else do_comp(); };

        const auto w0 = std::chrono::steady_clock::now();
        do { reset_X(); q->wait(); go(); q->wait(); }
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s());
        std::vector<double> ms;
        for (int r = 0; r < reps; ++r) {
            reset_X(); q->wait();
            const auto t0 = std::chrono::steady_clock::now();
            go();
            q->wait();
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        const Stat s = stat_of(ms);
        if (brk("sol"))
            for (int b : {0, batch - 1})
                for (size_t i = 0; i < sb; ++i)
                    R[size_t(b) * sb + i] = R[size_t(b) * sb + i] * T(1.01);
        const double res = solve_probe<T>(R, B0, A0, n, nrhs, batch, nprobe());
        const double gf = double(batch) * 2.0 * dn * dn * double(nrhs);
        const size_t ws = (mode == "getrs") ? s_ws
                            : (gather ? sizeof(T) * sb * batch + sizeof(int) * size_t(n) * batch : 0);
        const int ntp = nontrivial_pivots(pivi, piv_stride_i32, n);
        std::printf("%s,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.2f,%.3e,%zu,%s,%.3e,%d,%s\n",
                    mode.c_str(), tn, n, nrhs, batch, s.med, s.mean, s.relsd,
                    gf / (s.med * 1e6), res, ws, rt.c_str(), fres, ntp,
                    std::isfinite(res) && std::isfinite(fres) && ntp > 0 ? "ok" : "BAD");
        return 0;
    }

    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr,
            "usage: lubench <mode> <type> <n> <nrhs> <batch> <reps>\n"
            "modes : getrf | getrs | getrs_trsm | getri | getri_trsm\n"
            "types : float double cfloat cdouble\n"
            "cols  : op,type,n,nrhs,batch,med_ms,mean_ms,relsd,GFLOPs,resid,ws_bytes,route,extra,ntpiv,flag\n");
        return 2;
    }
    const std::string mode = argv[1], t = argv[2];
    const int n = std::atoi(argv[3]), nrhs = std::atoi(argv[4]);
    const int b = std::atoi(argv[5]), r = std::atoi(argv[6]);
    try {
        if (t == "float")   return run<float>(mode, "float", n, nrhs, b, r);
        if (t == "double")  return run<double>(mode, "double", n, nrhs, b, r);
        if (t == "cfloat")  return run<std::complex<float>>(mode, "cfloat", n, nrhs, b, r);
        if (t == "cdouble") return run<std::complex<double>>(mode, "cdouble", n, nrhs, b, r);
    } catch (const std::exception& e) {
        std::printf("%s,%s,%d,%d,%d,THREW,%s\n", mode.c_str(), t.c_str(), n, nrhs, b, e.what());
        return 0;
    }
    std::fprintf(stderr, "unknown type %s\n", t.c_str());
    return 2;
}
