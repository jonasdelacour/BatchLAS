// WP4 Phase 2 BENCHMARK -- the blocked native potrf against cuSOLVER.
//
// WHY THIS AND NOT A SYNTHETIC gemm/trsm HARNESS: every operand the blocked
// driver hands gemm and trsm is a SUB-VIEW of the parent carrying the PARENT ld
// and the PARENT batch stride.  The native GEMM fast paths are gated on
// is_contiguous_dense_matrix (register_tiled_common.hh:74-77), which such a
// sub-view fails by construction.  WP3 measured 0.86-0.98x of cuBLAS at
// ld == rows and 0.43-0.62x at the real ld on the SAME shapes.  Only potrf
// itself issues those views, so only potrf can measure the effect.
//
// WHAT IS COMPARED, in ONE process, INTERLEAVED, on the SAME parent allocation:
//
//   vendor   backend::potrf_vendor<CUDA,T>          -- cuSOLVER, the target.
//   blocked  sycl_potrf::potrf_blocked_dispatch<T> with the routed gemm and the
//            routed trsm injected -- i.e. BYTE-FOR-BYTE what the facade's
//            Algorithm::Blocked arm does (factorization.cc:261-276).  Whether
//            those injected calls land on the vendor or on the native kernel is
//            then decided by BATCHLAS_GEMM_ROUTE / BATCHLAS_TRSM_ROUTE, which is
//            how one binary measures both the vendor-present blocked route and
//            the vendor-free one.
//   cta      sycl_potrf::potrf_cta_dispatch<T>      -- only when n fits.
//
// A direct call is used rather than the facade for the reason potrf_native.hh
// gives: a forced route that supports() rejects falls back to automatic()
// (route_resolve.hh:101,:111) and SILENTLY runs cuSOLVER, so an env-pinned
// benchmark can be timing the vendor while believing it is timing the kernel.
// A direct call cannot be served by a vendor.  Mode `facade` exists to check
// that the pinned facade and the direct call agree, and mode `route` prints what
// the resolver actually returns for the pin.
//
// Modes:
//   route  <type> <n> <batch>              resolved Route for potrf, printed.
//   ab     <type> <n> <batch> <reps>       the measurement.  CSV to stdout.
//   facade <type> <n> <batch> <reps>       the same, through the public facade.
//
// env: BATCHLAS_GEMM_ROUTE, BATCHLAS_TRSM_ROUTE, BATCHLAS_POTRF_ROUTE,
//      BATCHLAS_POTRF_NB, BATCHLAS_POTRF_W  (all read by the library),
//      BENCH_WARM_S (default 2.0), BENCH_CHECK_COLS (default 4).
#include <batchlas/blas/linalg.hh>
#include <batchlas/sycl_interop.hh>
#include <batchlas/backend_config.h>
#include "../../../src/extensions/potrf_native.hh"
#include "../../../src/backends/potrf_route.hh"
#include <batchlas/blas/dispatch/vendor_available.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace batchlas;

// NOT BATCHLAS_HAS_CUDA_BACKEND.  build-novendor still has the CUDA backend --
// it is the vendor LIBRARY that is absent -- so that macro is 1 there and the
// vendor arm fails to link.  This is the compile-time fact route_potrf.hh:359
// names.
static constexpr bool kVendorSolver = dispatch::solver_vendor_available<Backend::CUDA>;

static double env_dbl(const char* v, double d) {
    if (const char* p = std::getenv(v)) return std::atof(p);
    return d;
}

template <typename T> struct RealOf { using type = T; };
template <typename U> struct RealOf<std::complex<U>> { using type = U; };

static float  conj_of(float v)  { return v; }
static double conj_of(double v) { return v; }
static std::complex<float>  conj_of(std::complex<float> v)  { return std::conj(v); }
static std::complex<double> conj_of(std::complex<double> v) { return std::conj(v); }
static double abs_of(float v)  { return std::fabs(double(v)); }
static double abs_of(double v) { return std::fabs(v); }
static double abs_of(std::complex<float> v)  { return std::abs(std::complex<double>(v.real(), v.imag())); }
static double abs_of(std::complex<double> v) { return std::abs(v); }
static bool finite_of(float v)  { return std::isfinite(v); }
static bool finite_of(double v) { return std::isfinite(v); }
static bool finite_of(std::complex<float> v)  { return std::isfinite(v.real()) && std::isfinite(v.imag()); }
static bool finite_of(std::complex<double> v) { return std::isfinite(v.real()) && std::isfinite(v.imag()); }

struct Rng {
    uint32_t s = 0x9E3779B9u;
    double next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5;
                    return double(s) / 4294967296.0 - 0.5; }
};
template <typename T> static T from_rng(Rng& r);
template <> float from_rng<float>(Rng& r) { return float(r.next()); }
template <> double from_rng<double>(Rng& r) { return r.next(); }
template <> std::complex<float> from_rng<std::complex<float>>(Rng& r) {
    const float a = float(r.next()); const float b = float(r.next());
    return std::complex<float>(a, b);
}
template <> std::complex<double> from_rng<std::complex<double>>(Rng& r) {
    const double a = r.next(); const double b = r.next();
    return std::complex<double>(a, b);
}

// A Hermitian random off-diagonal pattern built ONCE, O(n^2), plus a diagonal
// of n per item.  Diagonal dominance is the point: the off-diagonal part has
// entries in [-0.5,0.5] so its spectral radius is ~0.6*sqrt(n) (26 at n=2048)
// against a diagonal of n, giving a condition number under 1.05.  NO correctly
// implemented Cholesky can fail on this, which is what makes info != 0 out of
// this input a DRIVER defect and never a property of the matrix -- and the
// vendor arm, run on the identical buffer in the identical process, is the
// standing control for that claim.
//
// It replaced an O(n^3) Gram fill (G = M^H M) purely for cost: 8.6 GFLOP of
// host triple loop at n=2048 is a minute per process.  The failures reported
// below were first found under the Gram fill and reproduce identically here, so
// nothing about them is an artefact of the input.
//
// The whole allocation is poisoned first so a read outside the named window
// yields garbage rather than a plausible answer.
template <typename T>
static void fill_hpd(T* p, int n, int ld, int stride, int batch, size_t total) {
    Rng r;
    for (size_t i = 0; i < total; ++i) p[i] = from_rng<T>(r);
    std::vector<T> G(size_t(n) * size_t(n));
    {
        Rng rb; rb.s = 0x9E3779B9u;
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i) {
                const T v = (i == j) ? T(0) : from_rng<T>(rb);
                G[size_t(j) * size_t(n) + size_t(i)] = v;
                G[size_t(i) * size_t(n) + size_t(j)] = conj_of(v);
            }
    }
    for (int b = 0; b < batch; ++b) {
        T* A = p + size_t(b) * size_t(stride);
        const double shift = double(n) * (1.0 + 0.01 * double(b % 17));
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i)
                A[size_t(j) * size_t(ld) + size_t(i)] = G[size_t(j) * size_t(n) + size_t(i)];
            A[size_t(j) * size_t(ld) + size_t(j)] = A[size_t(j) * size_t(ld) + size_t(j)] + T(shift);
        }
    }
}

// [FIX-B-trap]: explicit 6-arg ctor, PARENT ld AND stride AND batch, never
// operator()(Slice,Slice) (matrix.hh:1140 propagates the parent pointer array)
// and never a defaulted stride (matrix.cc:1839 resolves 0 to ld*cols of the
// CHILD, which breaks every batch item after the first).  `ptrs` is this view's
// OWN pointer array -- the vendor batched call does A.data_ptrs(ctx), which
// throws on a view built without one.
template <typename T>
static MatrixView<T, MatrixFormat::Dense> viewof(T* base, int nr, int nc,
                                                 int ld, int stride, int batch,
                                                 T** ptrs) {
    return MatrixView<T, MatrixFormat::Dense>(base, nr, nc, ld, stride, batch, ptrs);
}

struct Stats { double med, mn, rel_sd; };
static Stats summarize(std::vector<double> v) {
    std::vector<double> s = v; std::sort(s.begin(), s.end());
    double mean = 0; for (double x : v) mean += x; mean /= double(v.size());
    double var = 0; for (double x : v) var += (x - mean) * (x - mean);
    return {s[s.size()/2], s.front(), std::sqrt(var / double(v.size())) / mean};
}

static const char* orig_name(dispatch::Origin o) {
    switch (o) { case dispatch::Origin::Auto: return "Auto";
                 case dispatch::Origin::Native: return "Native";
                 case dispatch::Origin::Vendor: return "Vendor"; }
    return "?";
}
static const char* algo_name(dispatch::Algorithm a) {
    switch (a) { case dispatch::Algorithm::Auto: return "Auto";
                 case dispatch::Algorithm::Direct: return "Direct";
                 case dispatch::Algorithm::RegisterTiled: return "RegisterTiled";
                 case dispatch::Algorithm::CTA: return "CTA";
                 case dispatch::Algorithm::Blocked: return "Blocked";
                 default: return "other"; }
}

// Residual on a COLUMN SUBSET.  ||A(:,c) - (L L^H)(:,c)|| / (n * ||A(:,c)||)
// over the lower part only, for a few evenly spaced c.  O(n^2) per column
// instead of O(n^3) for the whole matrix, which is what makes it affordable to
// check EVERY timed configuration rather than a sampled one -- a benchmark that
// only checks correctness in a separate run can time a wrong answer.
template <typename T>
static double residual_cols(const T* Lp, const T* Ap, int n, int ld, int ncols) {
    using R = typename RealOf<T>::type;
    double worst = 0.0;
    for (int s = 0; s < ncols; ++s) {
        const int c = (ncols == 1) ? 0 : int(double(s) * double(n - 1) / double(ncols - 1) + 0.5);
        double num = 0.0, den = 0.0;
        for (int i = c; i < n; ++i) {
            T acc = T(0);
            const int tmax = std::min(i, c);
            for (int t = 0; t <= tmax; ++t)
                acc += Lp[size_t(t) * size_t(ld) + size_t(i)]
                     * conj_of(Lp[size_t(t) * size_t(ld) + size_t(c)]);
            const T a = Ap[size_t(c) * size_t(ld) + size_t(i)];
            const double d = abs_of(T(acc - a));
            num += d * d;
            den += abs_of(a) * abs_of(a);
        }
        const double r = std::sqrt(num) / (double(n) * std::sqrt(den) + 1e-300);
        worst = std::max(worst, r);
        (void)sizeof(R);
    }
    return worst;
}

// Words of the strict UPPER triangle that changed.  LAPACK potrf(Lower) must
// leave them alone, and the measure phase recorded that a broken trailing
// update (a plain square gemm over A22) leaves the RESIDUAL green and is caught
// by nothing else.
template <typename T>
static long upper_changed(const T* Lp, const T* Ap, int n, int ld) {
    long c = 0;
    for (int j = 1; j < n; ++j)
        for (int i = 0; i < j; ++i) {
            const size_t o = size_t(j) * size_t(ld) + size_t(i);
            if (std::memcmp(&Lp[o], &Ap[o], sizeof(T)) != 0) ++c;
        }
    return c;
}

template <typename T>
static long nonfinite_count(const T* Lp, int n, int ld, int stride, int batch) {
    // First 16 items and the last one -- a full scan is O(n^2 * batch) on the
    // host and costs seconds per cell at n=2048.  info != 0 is the primary
    // non-finiteness detector; this is the backstop for a quench that let a
    // NaN through, and the last item is included because a stride defect that
    // spares item 0 lands there.
    long c = 0;
    const int lim = std::min(batch, 16);
    for (int b = 0; b < batch; ++b) {
        if (b >= lim && b != batch - 1) continue;
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i)
                if (!finite_of(Lp[size_t(b) * size_t(stride) + size_t(j) * size_t(ld) + size_t(i)])) ++c;
    }
    return c;
}

// ---------------------------------------------------------------------------

enum class Variant { Vendor, Blocked, Cta, Facade };
static const char* variant_name(Variant v) {
    switch (v) { case Variant::Vendor: return "vendor";
                 case Variant::Blocked: return "blocked";
                 case Variant::Cta: return "cta";
                 case Variant::Facade: return "facade"; }
    return "?";
}

template <typename T>
static int run_ab(const char* tn, int n, int batch, int reps, bool use_facade) {
    auto q = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    sycl::queue& sq = sycl_queue(*q);

    const int ld = n, stride = n * n;
    const size_t total = size_t(stride) * size_t(batch);

    UnifiedVector<T> Abuf(total);       // the working copy, factorised in place
    UnifiedVector<T> Pbuf(total);       // pristine, restored from before each call
    const size_t nbatch = static_cast<size_t>(batch);
    UnifiedVector<T*> ptrs(nbatch);
    UnifiedVector<int32_t> info(nbatch);

    fill_hpd<T>(Pbuf.data(), n, ld, stride, batch, total);

    auto A = viewof<T>(Abuf.data(), n, n, ld, stride, batch, ptrs.data());

    // Workspace: the MAX over both routes, so one lease serves every variant and
    // no variant is measured with a different allocation than another.
    size_t wsz = sycl_potrf::potrf_blocked_buffer_size<T>(*q, A, Uplo::Lower);
    wsz = std::max(wsz, sycl_potrf::potrf_cta_buffer_size<T>(*q, A));
    if constexpr (kVendorSolver)
        wsz = std::max(wsz, backend::potrf_vendor_buffer_size<Backend::CUDA, T>(*q, A, Uplo::Lower));
    wsz = std::max(wsz, size_t(1));
    UnifiedVector<std::byte> ws(wsz);

    const auto restore = [&] {
        sq.memcpy(Abuf.data(), Pbuf.data(), total * sizeof(T)).wait();
    };

    const int cta_max = sycl_potrf::potrf_cta_max_n<T>();
    const unsigned bp = sycl_potrf::potrf_blocked_debug_params<T>(*q, n);
    const int nb = int(bp & 0xFFFFu), W = int(bp >> 16);

    std::vector<Variant> variants;
    if constexpr (kVendorSolver) variants.push_back(Variant::Vendor);
    variants.push_back(use_facade ? Variant::Facade : Variant::Blocked);
    if (!use_facade && n <= cta_max) variants.push_back(Variant::Cta);

    const auto call = [&](Variant v) {
        switch (v) {
            case Variant::Vendor:
                if constexpr (kVendorSolver)
                    backend::potrf_vendor<Backend::CUDA, T>(*q, A, Uplo::Lower, ws.to_span(), info.to_span());
                break;
            case Variant::Facade:
                potrf<Backend::CUDA, T>(*q, A, Uplo::Lower, ws.to_span(), info.to_span());
                break;
            case Variant::Cta:
                sycl_potrf::potrf_cta_dispatch<T>(*q, A, Uplo::Lower, ws.to_span(), info.to_span());
                break;
            case Variant::Blocked:
                // EXACTLY factorization.cc:261-276.  The routed gemm and the
                // routed trsm, injected -- not gemm_custom, which is the WP3
                // step 16 recorded defect.
                sycl_potrf::potrf_blocked_dispatch<T>(
                    *q, A, Uplo::Lower, ws.to_span(), info.to_span(),
                    [](Queue& c, const MatrixView<T, MatrixFormat::Dense>& ga,
                       const MatrixView<T, MatrixFormat::Dense>& gb,
                       const MatrixView<T, MatrixFormat::Dense>& gc,
                       T galpha, T gbeta, Transpose gta, Transpose gtb,
                       ComputePrecision gp) {
                        return gemm<Backend::CUDA, T>(c, ga, gb, gc, galpha, gbeta, gta, gtb, gp);
                    },
                    [](Queue& c, const MatrixView<T, MatrixFormat::Dense>& ta,
                       const MatrixView<T, MatrixFormat::Dense>& tb,
                       T talpha, Side tside, Uplo tuplo, Transpose ttrans, Diag tdiag) {
                        return trsm<Backend::CUDA, T>(c, ta, tb, talpha, tside, tuplo, ttrans, tdiag);
                    });
                break;
        }
    };

    // CORRECTNESS BEFORE ANY TIMING, per variant, on the very shape about to be
    // timed.  Never in a separate run: the point is that no timed cell in the
    // CSV can be a wrong answer.
    const int ncols = int(env_dbl("BENCH_CHECK_COLS", 4));
    std::vector<double> resid(variants.size(), -1.0);
    std::vector<long>   upch(variants.size(), -1);
    std::vector<long>   nonfin(variants.size(), -1);
    std::vector<int>    infonz(variants.size(), -1);
    for (size_t vi = 0; vi < variants.size(); ++vi) {
        restore();
        for (int b = 0; b < batch; ++b) info[b] = -12345;
        call(variants[vi]);
        q->wait();
        int nz = 0;
        for (int b = 0; b < batch; ++b) if (info[b] != 0) ++nz;
        infonz[vi] = nz;
        if (nz > 0) {
            // Say WHICH items and WHAT value.  A bare count cannot distinguish
            // "the driver never wrote info" (-12345, the poison) from a real
            // LAPACK-positive leading-minor report, and those are different bugs.
            std::fprintf(stderr, "INFONZ %s %s n=%d batch=%d nz=%d:", variant_name(variants[vi]), tn, n, batch, nz);
            int shown = 0;
            for (int b = 0; b < batch && shown < 8; ++b)
                if (info[b] != 0) { std::fprintf(stderr, " [%d]=%d", b, info[b]); ++shown; }
            // and the residual of the FIRST failing item, which items 0 and
            // batch-1 can easily both miss.
            for (int b = 0; b < batch; ++b)
                if (info[b] != 0) {
                    const double rr = residual_cols<T>(Abuf.data() + size_t(b) * size_t(stride),
                                                       Pbuf.data() + size_t(b) * size_t(stride),
                                                       n, ld, ncols);
                    std::fprintf(stderr, " resid[%d]=%.3e", b, rr);
                    break;
                }
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
        }
        // item 0 and item batch-1: the second is what catches a stride bug that
        // leaves item 0 perfect.
        double r = residual_cols<T>(Abuf.data(), Pbuf.data(), n, ld, ncols);
        r = std::max(r, residual_cols<T>(Abuf.data() + size_t(batch - 1) * size_t(stride),
                                         Pbuf.data() + size_t(batch - 1) * size_t(stride),
                                         n, ld, ncols));
        resid[vi] = r;
        upch[vi] = upper_changed<T>(Abuf.data(), Pbuf.data(), n, ld);
        nonfin[vi] = nonfinite_count<T>(Abuf.data(), n, ld, stride, batch);
    }

    // WARM.  A first-run SYCL JIT once fabricated a 3.7x loss in this tree, and
    // an idle 4090 sits at 210 MHz until it is asked to do something.  Every
    // variant is warmed, and warmed AFTER the correctness pass so the clocks are
    // already up when the first one starts.
    const double warm_s = env_dbl("BENCH_WARM_S", 2.0);
    {
        auto w0 = std::chrono::steady_clock::now();
        do {
            for (Variant v : variants) { restore(); call(v); }
            q->wait();
        } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s);
    }

    // INTERLEAVED.  All of A then all of B lets a drifting clock or a
    // background process land entirely on one arm.
    std::vector<std::vector<double>> ms(variants.size());
    for (int r = 0; r < reps; ++r) {
        for (size_t vi = 0; vi < variants.size(); ++vi) {
            restore();
            q->wait();
            auto t0 = std::chrono::steady_clock::now();
            call(variants[vi]);
            q->wait();
            auto t1 = std::chrono::steady_clock::now();
            ms[vi].push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }

    // n^3/3 flops for the factorisation, x4 for complex (a complex multiply-add
    // is 4 real multiplies and 4 real adds; the standard counting).
    const double base = double(n) * double(n) * double(n) / 3.0 * double(batch);
    const double flops = base * ((sizeof(T) == 2 * sizeof(typename RealOf<T>::type)) ? 4.0 : 1.0);

    for (size_t vi = 0; vi < variants.size(); ++vi) {
        const Stats s = summarize(ms[vi]);
        std::printf("%s,%s,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.2f,%.3e,%ld,%ld,%d\n",
                    variant_name(variants[vi]), tn, n, batch, nb, W,
                    s.med, s.mn, s.rel_sd, flops / (s.med * 1e-3) / 1e9,
                    resid[vi], upch[vi], nonfin[vi], infonz[vi]);
        std::fflush(stdout);
    }
    return 0;
}

template <typename T>
static int run_route(const char* tn, int n, int batch) {
    auto q = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    UnifiedVector<T> Abuf(size_t(n) * size_t(n) * size_t(batch));
    const size_t nbatch = static_cast<size_t>(batch);
    UnifiedVector<T*> ptrs(nbatch);
    auto A = viewof<T>(Abuf.data(), n, n, n, n * n, batch, ptrs.data());
    const bool va = kVendorSolver;
    const auto r = backend::potrf_route<Backend::CUDA, T>(*q, A, Uplo::Lower, va);
    const unsigned bp = sycl_potrf::potrf_blocked_debug_params<T>(*q, n);
    const char* e = std::getenv("BATCHLAS_POTRF_ROUTE");
    std::printf("route,%s,n=%d,batch=%d,env=%s,vendor_available=%d,resolved=%s:%s,nb=%u,W=%u,cta_max_n=%d\n",
                tn, n, batch, e ? e : "(unset)", va ? 1 : 0,
                orig_name(r.origin), algo_name(r.algo),
                bp & 0xFFFFu, bp >> 16, sycl_potrf::potrf_cta_max_n<T>());
    return 0;
}

template <typename F>
static int by_type(const char* t, F&& f) {
    if (!std::strcmp(t, "float"))   return f.template operator()<float>();
    if (!std::strcmp(t, "double"))  return f.template operator()<double>();
    if (!std::strcmp(t, "cfloat"))  return f.template operator()<std::complex<float>>();
    if (!std::strcmp(t, "cdouble")) return f.template operator()<std::complex<double>>();
    std::fprintf(stderr, "unknown type %s\n", t);
    return 2;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: bench route  <type> <n> <batch>\n"
            "       bench ab     <type> <n> <batch> <reps>\n"
            "       bench facade <type> <n> <batch> <reps>\n");
        return 2;
    }
    const std::string mode = argv[1];
    const char* tn = argc > 2 ? argv[2] : "float";
    const int n = argc > 3 ? std::atoi(argv[3]) : 512;
    const int batch = argc > 4 ? std::atoi(argv[4]) : 128;
    const int reps = argc > 5 ? std::atoi(argv[5]) : 5;

    try {
        if (mode == "route") {
            auto f = [&]<typename T>() { return run_route<T>(tn, n, batch); };
            return by_type(tn, f);
        }
        if (mode == "ab" || mode == "facade") {
            const bool fac = (mode == "facade");
            auto f = [&]<typename T>() { return run_ab<T>(tn, n, batch, reps, fac); };
            return by_type(tn, f);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION %s/%s n=%d batch=%d: %s\n",
                     mode.c_str(), tn, n, batch, e.what());
        return 3;
    }
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
