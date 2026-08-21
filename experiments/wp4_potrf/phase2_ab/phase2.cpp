// WP4 Phase 2 A/B harness -- open questions 5 and 6, plus the nb sweep.
//
// WHY A BESPOKE HARNESS AND NOT benchmarks/trsm_benchmark + gemm_benchmark:
//   * both allocate SQUARE, ld == rows operands. Every operand a blocked
//     Cholesky driver hands trsm/gemm is a SUB-VIEW of the parent carrying the
//     PARENT ld and the PARENT batch stride. A benchmark that allocates its own
//     operands is structurally incapable of seeing the strided-ld effect that
//     WP3 measured at 0.43-0.62x (see the `flat` vs `sub` modes below, which
//     differ ONLY in that).
//   * the split of a whole blocked potrf across leaf / panel / trailing cannot
//     be obtained from any per-op benchmark at all.
//
// Modes:
//   panel  <type> <n> <nb> <batch> <reps>
//        the panel solve trsm(Right,Lower,ConjTrans,NonUnit) for EVERY j the
//        blocked driver would issue, operands built as real sub-views.
//   panelflat <type> <m2> <ib> <batch> <reps>
//        the same shape but with freshly allocated ld == rows operands -- the
//        control that isolates "strided sub-view" from "shape".
//   trail  <type> <m> <n> <k> <batch> <sub|flat> <T|C> <reps>
//        one trailing-update gemm, alpha=-1, beta=1, transA=N.
//   blocked <type> <n> <nb> <W> <batch> <reps>
//        the whole right-looking blocked driver, per-stage timings, plus a
//        residual check before any timing.
//   vendorpotrf <type> <n> <batch> <reps>
//        the routed potrf (cuSOLVER when present) on the same parent, for scale.
//
// env: BATCHLAS_TRSM_ROUTE / BATCHLAS_GEMM_ROUTE / BATCHLAS_POTRF_ROUTE all
//      honoured by the library; BENCH_WARM_S (default 1.5).
#include <batchlas/blas/linalg.hh>
#include <batchlas/backend_config.h>
#include "../../../src/extensions/potrf_native.hh"
#include "../../../src/extensions/symmetric_product_fold.hh"
#include "../../../src/sycl/trsm_native.hh"

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

// Deliberate-break switch. This repository has a recorded blind-guard pattern
// (a test that cannot fail by construction, guarding the property that later
// broke), so the residual check in `blocked` is itself checked: PHASE2_BREAK
// injects a defect and the run must go RED.
//   conj   -- trailing gemm uses Transpose::Trans instead of ConjTrans
//             (a no-op for real types by definition; must break complex).
//   nofold -- diagonal WxW block gemm written straight into A instead of into
//             scratch + fold (writes the upper triangle too).
//   stride -- every sub-view built with stride = ld*cols of the CHILD, i.e. the
//             matrix.cc:1839 default the [FIX-B-trap] warns about.
static const char* g_break = nullptr;
static bool broken(const char* w) { return g_break && std::strcmp(g_break, w) == 0; }

static double env_dbl(const char* v, double d) {
    if (const char* p = std::getenv(v)) return std::atof(p);
    return d;
}

template <typename T> struct RealOf { using type = T; };
template <typename U> struct RealOf<std::complex<U>> { using type = U; };

static float conj_of(float v) { return v; }
static double conj_of(double v) { return v; }
static std::complex<float> conj_of(std::complex<float> v) { return std::conj(v); }
static std::complex<double> conj_of(std::complex<double> v) { return std::conj(v); }

// Deterministic xorshift, same generator as experiments/wp4_complex/gpu1.
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

// Fill the whole allocation (padding included) with junk, then write an SPD/HPD
// matrix into the named n x n window at the given ld/stride. Junk in the pad is
// deliberate: it makes a driver that reads outside its window produce garbage
// rather than a plausible answer.
template <typename T>
static void fill_spd(T* p, int n, int ld, int stride, int batch, size_t total) {
    // Junk first, everywhere, so a driver that reads outside its named window
    // produces garbage rather than a plausible answer.
    Rng r;
    for (size_t i = 0; i < total; ++i) p[i] = from_rng<T>(r);

    // ONE Gram matrix G = M^H M, computed once (this is O(n^3) on the HOST --
    // at n=1024, batch=128 the per-item version cost ~30 s per process), then
    // replicated with a per-item diagonal shift so the items are not bitwise
    // identical. G + s*I with s >= n is diagonally dominant by a wide margin:
    // eig(G) in [0, 4n*var] = [0, n/3] for var = 1/12, so the condition number
    // is under 1.4 and no correctly implemented Cholesky can fail on it. That
    // matters: an `info != 0` from this input is a DRIVER defect, never a
    // property of the matrix.
    // PHASE2_FILL=peritem restores the ORIGINAL per-batch-item Gram fill, kept
    // so the failure it produced can be re-run against cuSOLVER: if the vendor
    // fails on the same input, the input was the defect, not the driver.
    const bool per_item = [] {
        const char* e = std::getenv("PHASE2_FILL");
        return e && std::strcmp(e, "peritem") == 0;
    }();
    std::vector<T> G(size_t(n) * size_t(n));
    {
        std::vector<T> M(size_t(n) * size_t(n));
        Rng rb; rb.s = 0x9E3779B9u;
        for (size_t i = 0; i < M.size(); ++i) M[i] = from_rng<T>(rb);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                T acc = T(0);
                for (int t = 0; t < n; ++t)
                    acc += conj_of(M[size_t(i) * size_t(n) + size_t(t)])
                         * M[size_t(j) * size_t(n) + size_t(t)];
                G[size_t(j) * size_t(n) + size_t(i)] = acc;
            }
    }
    for (int b = 0; b < batch; ++b) {
        if (per_item) {
            std::vector<T> M(size_t(n) * size_t(n));
            Rng rb; rb.s = 0x9E3779B9u + uint32_t(b) * 2654435761u;
            for (size_t i = 0; i < M.size(); ++i) M[i] = from_rng<T>(rb);
            T* Ab = p + size_t(b) * size_t(stride);
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    T acc = T(0);
                    for (int t = 0; t < n; ++t)
                        acc += conj_of(M[size_t(i)*size_t(n)+size_t(t)]) * M[size_t(j)*size_t(n)+size_t(t)];
                    Ab[size_t(j)*size_t(ld)+size_t(i)] = acc;
                }
                Ab[size_t(j)*size_t(ld)+size_t(j)] = Ab[size_t(j)*size_t(ld)+size_t(j)] + T(double(n));
            }
            continue;
        }
        T* A = p + size_t(b) * size_t(stride);
        const double shift = double(n) * (1.0 + 0.01 * double(b % 17));
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i)
                A[size_t(j) * size_t(ld) + size_t(i)] = G[size_t(j) * size_t(n) + size_t(i)];
            A[size_t(j) * size_t(ld) + size_t(j)] =
                A[size_t(j) * size_t(ld) + size_t(j)] + T(shift);
        }
    }
}

template <typename T>
static MatrixView<T, MatrixFormat::Dense> sub(T* base, int r0, int nr, int c0, int nc,
                                              int ld, int stride, int batch,
                                              T** ptrs = nullptr) {
    // [FIX-B-trap]: explicit 6-arg ctor, parent ld AND stride AND batch, never
    // operator()(Slice,Slice) (matrix.hh:1140 propagates the parent pointer
    // array) and never a defaulted stride (matrix.cc:1839 resolves 0 to ld*cols
    // of the CHILD, which breaks every batch item after the first).
    // `ptrs` is scratch for a per-view pointer array, NOT the parent's -- the
    // vendor batched trsm (cublas.cc:1221) calls A.data_ptrs(ctx), which throws
    // "data_ptrs target is null" on a view built without one. Each caller here
    // hands each ROLE its own array, and init_data_ptr_array recomputes it from
    // this view's own data_ptr()/stride on every call.
    return MatrixView<T, MatrixFormat::Dense>(
        base + std::ptrdiff_t(c0) * ld + r0, nr, nc, ld, stride, batch, ptrs);
}

struct Stats { double med, mn, rel_sd; };
static Stats summarize(std::vector<double> v) {
    std::vector<double> s = v; std::sort(s.begin(), s.end());
    double mean = 0; for (double x : v) mean += x; mean /= double(v.size());
    double var = 0; for (double x : v) var += (x - mean) * (x - mean);
    return {s[s.size()/2], s.front(), std::sqrt(var / double(v.size())) / mean};
}

// ---------------------------------------------------------------- panel solve
template <typename T>
static int mode_panel(const char* tn, int n, int nb, int batch, int reps, bool flat,
                      int flat_m2, int flat_ib) {
    auto q = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    const double warm_s = env_dbl("BENCH_WARM_S", 1.5);
    // The FIRST job pays the full warm-up (SYCL JIT once fabricated a 3.7x loss
    // in this tree, and an idle 4090 sits at 210 MHz). Later jobs in the same
    // process only need a short re-warm: the card has been busy throughout.
    const double rewarm_s = env_dbl("BENCH_REWARM_S", 0.25);
    const int max_jobs = int(env_dbl("PANEL_MAX_JOBS", 4));

    struct Job { int m2, ib, j; };
    std::vector<Job> all;
    if (flat) {
        all.push_back({flat_m2, flat_ib, 0});
    } else {
        for (int j = 0; j + nb < n; j += nb) {
            const int ib = std::min(nb, n - j);
            const int m2 = n - j - ib;
            if (m2 > 0) all.push_back({m2, ib, j});
        }
    }
    // Subsample evenly rather than truncate: the panel shape sweeps from
    // m2 = n-nb down to m2 = nb, and the ends are the interesting cells.
    std::vector<Job> jobs;
    if (int(all.size()) <= max_jobs) {
        jobs = all;
    } else {
        for (int i = 0; i < max_jobs; ++i)
            jobs.push_back(all[size_t(double(i) * double(all.size()-1) / double(max_jobs-1) + 0.5)]);
    }

    // ONE parent allocation for the whole sweep, and the parent is the same
    // n x n buffer for every j -- which is what makes these sub-views the real
    // thing rather than a same-shaped stand-in.
    const int Pld     = flat ? std::max(jobs[0].ib, 1) : n;
    const int Pstride = flat ? Pld * jobs[0].ib : n * n;
    UnifiedVector<T> Abuf(size_t(Pstride) * size_t(batch));
    UnifiedVector<T> Bbuf(flat ? size_t(jobs[0].m2) * size_t(jobs[0].ib) * size_t(batch) : size_t(1));
    T* Ap = Abuf.data();
    {
        Rng r;
        for (size_t i = 0; i < Abuf.size(); ++i) Ap[i] = from_rng<T>(r);
        const int order = flat ? jobs[0].ib : n;
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < order; ++c)
                Ap[size_t(b)*size_t(Pstride) + size_t(c)*size_t(Pld) + size_t(c)] = T(double(order));
    }
    T* Bp = flat ? Bbuf.data() : Ap;
    if (flat) { Rng r; for (size_t i = 0; i < Bbuf.size(); ++i) Bp[i] = from_rng<T>(r); }

    UnifiedVector<T*> pA(static_cast<size_t>(batch));
    UnifiedVector<T*> pB(static_cast<size_t>(batch));

    bool first = true;
    for (const Job& jb : jobs) {
        const int Bld     = flat ? jb.m2 : n;
        const int Bstride = flat ? jb.m2 * jb.ib : n * n;
        auto A11 = flat ? sub<T>(Ap, 0, jb.ib, 0, jb.ib, Pld, Pstride, batch, pA.data())
                        : sub<T>(Ap, jb.j, jb.ib, jb.j, jb.ib, n, n*n, batch, pA.data());
        auto A21 = flat ? sub<T>(Bp, 0, jb.m2, 0, jb.ib, Bld, Bstride, batch, pB.data())
                        : sub<T>(Ap, jb.j + jb.ib, jb.m2, jb.j, jb.ib, n, n*n, batch, pB.data());

        auto call = [&] {
            trsm(*q, A11, A21, T(1), Side::Right, Uplo::Lower,
                 Transpose::ConjTrans, Diag::NonUnit);
        };
        const double ws = first ? warm_s : rewarm_s;
        first = false;
        auto w0 = std::chrono::steady_clock::now();
        do { for (int i = 0; i < 3; ++i) call(); q->wait(); }
        while (std::chrono::duration<double>(std::chrono::steady_clock::now()-w0).count() < ws);

        std::vector<double> ms;
        for (int r = 0; r < reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            const int inner = 3;
            for (int i = 0; i < inner; ++i) call();
            q->wait();
            auto t1 = std::chrono::steady_clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(t1-t0).count()/inner);
        }
        Stats s = summarize(ms);
        const double fs = (std::string(tn).front() == 'c') ? 4.0 : 1.0;
        const double gf = fs * double(jb.m2) * double(jb.ib) * double(jb.ib) * double(batch) / (s.med * 1e6);
        std::printf("%s,%s,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.4f,%.2f\n",
                    flat ? "panelflat" : "panel", tn, n, nb, jb.j, jb.m2, jb.ib, batch,
                    s.med, s.mn, s.rel_sd, gf);
        std::fflush(stdout);
    }
    return 0;
}

// ------------------------------------------------------------ trailing update
template <typename T>
static int mode_trail(const char* tn, int m, int nn, int k, int batch,
                      bool flat, Transpose tB, int reps) {
    auto q = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    const double warm_s = env_dbl("BENCH_WARM_S", 1.5);
    const int P = std::max(m, nn) + k;
    const int ld     = flat ? 1 : P;
    const int stride = flat ? 1 : P * P;

    UnifiedVector<T> parent(flat ? size_t(1) : size_t(stride) * size_t(batch));
    UnifiedVector<T> Af(flat ? size_t(m) * size_t(k) * size_t(batch) : size_t(1));
    UnifiedVector<T> Bf(flat ? size_t(nn) * size_t(k) * size_t(batch) : size_t(1));
    UnifiedVector<T> Cf(flat ? size_t(m) * size_t(nn) * size_t(batch) : size_t(1));
    Rng r;
    auto fill = [&](UnifiedVector<T>& v) {
        for (size_t i = 0; i < v.size(); ++i) v.data()[i] = from_rng<T>(r);
    };
    if (flat) { fill(Af); fill(Bf); fill(Cf); } else { fill(parent); }

    MatrixView<T, MatrixFormat::Dense> A = flat
        ? sub<T>(Af.data(), 0, m, 0, k, m, m*k, batch)
        : sub<T>(parent.data(), k, m, 0, k, ld, stride, batch);
    MatrixView<T, MatrixFormat::Dense> B = flat
        ? sub<T>(Bf.data(), 0, nn, 0, k, nn, nn*k, batch)
        : sub<T>(parent.data(), k, nn, 0, k, ld, stride, batch);
    MatrixView<T, MatrixFormat::Dense> C = flat
        ? sub<T>(Cf.data(), 0, m, 0, nn, m, m*nn, batch)
        : sub<T>(parent.data(), k, m, k, nn, ld, stride, batch);

    auto call = [&] { gemm(*q, A, B, C, T(-1), T(1), Transpose::NoTrans, tB); };
    auto w0 = std::chrono::steady_clock::now();
    do { for (int i = 0; i < 3; ++i) call(); q->wait(); }
    while (std::chrono::duration<double>(std::chrono::steady_clock::now()-w0).count() < warm_s);

    std::vector<double> ms;
    for (int rr = 0; rr < reps; ++rr) {
        auto t0 = std::chrono::steady_clock::now();
        const int inner = 3;
        for (int i = 0; i < inner; ++i) call();
        q->wait();
        auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1-t0).count()/inner);
    }
    Stats s = summarize(ms);
    const double fs = (std::string(tn).front() == 'c') ? 8.0 : 2.0;
    const double gf = fs * double(m) * double(nn) * double(k) * double(batch) / (s.med * 1e6);
    std::printf("trail,%s,%d,%d,%d,%d,%s,%c,%.6f,%.6f,%.4f,%.2f\n",
                tn, m, nn, k, batch, flat ? "flat" : "sub",
                tB == Transpose::Trans ? 'T' : 'C', s.med, s.mn, s.rel_sd, gf);
    std::fflush(stdout);
    return 0;
}

// -------------------------------------------------------- the blocked driver
template <typename T>
struct Blocked {
    Queue& q;
    int n, nb, W, batch;
    T* Ap;
    T* scratch;
    Span<int32_t> info;
    T** pA11 = nullptr;
    T** pA21 = nullptr;
    double t_leaf = 0, t_panel = 0, t_trail = 0;
    bool timed = false;

    void tick(std::chrono::steady_clock::time_point& t, double& acc) {
        if (!timed) return;
        q.wait();
        auto now = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double, std::milli>(now - t).count();
        t = now;
    }

    void run() {
        const int ld = n, st = n * n;
        auto t = std::chrono::steady_clock::now();
        for (int j = 0; j < n; j += nb) {
            const int ib = std::min(nb, n - j);
            const int stL = broken("stride") ? ld * ib : st;
            auto A11 = sub<T>(Ap, j, ib, j, ib, ld, stL, batch, pA11);
            sycl_potrf::potrf_cta_dispatch<T>(q, A11, Uplo::Lower, Span<std::byte>(), info);
            tick(t, t_leaf);
            const int m2 = n - j - ib;
            if (m2 == 0) break;
            auto A21 = sub<T>(Ap, j + ib, m2, j, ib, ld, st, batch, pA21);
            trsm(q, A11, A21, T(1), Side::Right, Uplo::Lower,
                 Transpose::ConjTrans, Diag::NonUnit);
            tick(t, t_panel);
            for (int c = 0; c < m2; c += W) {
                const int w = std::min(W, m2 - c);
                auto Lrow = sub<T>(Ap, j + ib + c, w, j, ib, ld, st, batch);
                auto Cd   = sub<T>(Ap, j + ib + c, w, j + ib + c, w, ld, st, batch);
                auto Sc   = sub<T>(scratch, 0, w, 0, w, W, W * W, batch);
                const Transpose tB = broken("conj") ? Transpose::Trans : Transpose::ConjTrans;
                if (broken("nofold")) {
                    gemm(q, Lrow, Lrow, Cd, T(-1), T(1), Transpose::NoTrans, tB);
                } else {
                    gemm(q, Lrow, Lrow, Sc, T(-1), T(0), Transpose::NoTrans, tB);
                    detail::fold_symmetric_product_into_triangle<T>(q, Cd, Sc, T(1), Uplo::Lower);
                }
                const int mr = m2 - c - w;
                if (mr > 0) {
                    auto Lr = sub<T>(Ap, j + ib + c + w, mr, j, ib, ld, st, batch);
                    auto Cr = sub<T>(Ap, j + ib + c + w, mr, j + ib + c, w, ld, st, batch);
                    gemm(q, Lr, Lrow, Cr, T(-1), T(1), Transpose::NoTrans, tB);
                }
            }
            tick(t, t_trail);
        }
        q.wait();
    }
};

template <typename T>
static double residual(const T* L, const T* A0, int n, int ld, int stride, int b) {
    using R = typename RealOf<T>::type;
    R num = 0, den = 0;
    const T* Lb = L + size_t(b) * size_t(stride);
    const T* Ab = A0 + size_t(b) * size_t(stride);
    for (int j = 0; j < n; ++j)
        for (int i = j; i < n; ++i) {
            T acc = T(0);
            for (int t = 0; t <= j; ++t)
                acc += Lb[size_t(t)*size_t(ld) + size_t(i)] * conj_of(Lb[size_t(t)*size_t(ld) + size_t(j)]);
            const T d = acc - Ab[size_t(j)*size_t(ld) + size_t(i)];
            num = std::max(num, R(std::abs(d)));
            den = std::max(den, R(std::abs(Ab[size_t(j)*size_t(ld) + size_t(i)])));
        }
    return double(num) / double(den);
}

template <typename T>
static int mode_blocked(const char* tn, int n, int nb, int W, int batch, int reps) {
    auto q = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    const double warm_s = env_dbl("BENCH_WARM_S", 1.5);
    const int ld = n, st = n * n;
    UnifiedVector<T> A(size_t(st) * size_t(batch)), A0(size_t(st) * size_t(batch));
    UnifiedVector<T> scratch(size_t(W) * size_t(W) * size_t(batch));
    UnifiedVector<int32_t> info(size_t(batch), 0);
    fill_spd<T>(A0.data(), n, ld, st, batch, A0.size());
    std::memcpy(A.data(), A0.data(), A.size() * sizeof(T));

    auto Aview  = sub<T>(A.data(),  0, n, 0, n, ld, st, batch);
    auto A0view = sub<T>(A0.data(), 0, n, 0, n, ld, st, batch);

    UnifiedVector<T*> pA11(static_cast<size_t>(batch));
    UnifiedVector<T*> pA21(static_cast<size_t>(batch));
    Blocked<T> bl{*q, n, nb, W, batch, A.data(), scratch.data(), info.to_span(),
                  pA11.data(), pA21.data()};

    bl.run();
    // Batch item 0 sits at offset 0, so a wrong sub-view STRIDE cannot move it
    // -- checking only item 0 is blind to exactly the [FIX-B-trap] defect this
    // harness is here to avoid. Take the worse of item 0 and item batch-1.
    const double res = std::max(residual<T>(A.data(), A0.data(), n, ld, st, 0),
                                residual<T>(A.data(), A0.data(), n, ld, st, batch - 1));
    int bad = 0; for (int i = 0; i < batch; ++i) if (info[size_t(i)] != 0) ++bad;
    if (std::getenv("PHASE2_DIAG")) {
        std::fprintf(stderr, "diag: info!=0 items:");
        for (int i = 0; i < batch; ++i)
            if (info[size_t(i)] != 0) std::fprintf(stderr, " %d(%d)", i, int(info[size_t(i)]));
        std::fprintf(stderr, "\n");
        for (int i = 0; i < std::min(batch, 8); ++i)
            std::fprintf(stderr, "diag: res[%d] = %.3e\n", i,
                         residual<T>(A.data(), A0.data(), n, ld, st, i));
    }

    auto reset = [&] { MatrixView<T, MatrixFormat::Dense>::copy(*q, Aview, A0view); q->wait(); };

    reset();
    auto w0 = std::chrono::steady_clock::now();
    do { bl.run(); reset(); }
    while (std::chrono::duration<double>(std::chrono::steady_clock::now()-w0).count() < warm_s);

    std::vector<double> tot;
    bl.timed = false;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        bl.run();
        auto t1 = std::chrono::steady_clock::now();
        tot.push_back(std::chrono::duration<double, std::milli>(t1-t0).count());
        reset();
    }
    Stats s = summarize(tot);

    bl.timed = true;
    double lf = 0, pn = 0, tr = 0, sy = 0;
    const int sreps = std::max(3, reps / 2);
    for (int r = 0; r < sreps; ++r) {
        bl.t_leaf = bl.t_panel = bl.t_trail = 0;
        auto t0 = std::chrono::steady_clock::now();
        bl.run();
        auto t1 = std::chrono::steady_clock::now();
        lf += bl.t_leaf; pn += bl.t_panel; tr += bl.t_trail;
        sy += std::chrono::duration<double, std::milli>(t1-t0).count();
        reset();
    }
    lf /= sreps; pn /= sreps; tr /= sreps; sy /= sreps;

    const double fs = (std::string(tn).front() == 'c') ? 4.0 : 1.0;
    const double gf = fs * (double(n)*double(n)*double(n)/3.0) * double(batch) / (s.med * 1e6);
    std::printf("blocked,%s,%d,%d,%d,%d,%.6f,%.6f,%.4f,%.2f,%.6f,%.6f,%.6f,%.6f,%.3e,%d\n",
                tn, n, nb, W, batch, s.med, s.mn, s.rel_sd, gf, sy, lf, pn, tr, res, bad);
    std::fflush(stdout);
    return 0;
}

template <typename T>
static int mode_vendorpotrf(const char* tn, int n, int batch, int reps) {
    auto q = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    const double warm_s = env_dbl("BENCH_WARM_S", 1.5);
    const int ld = n, st = n * n;
    UnifiedVector<T> A(size_t(st) * size_t(batch)), A0(size_t(st) * size_t(batch));
    UnifiedVector<int32_t> info(size_t(batch), 0);
    fill_spd<T>(A0.data(), n, ld, st, batch, A0.size());
    // cusolverDnXpotrfBatched takes a POINTER ARRAY, and a MatrixView built by
    // the 6-arg constructor has none -- data_ptrs(ctx) then throws
    // "data_ptrs target is null" (src/matrix.cc:2369). Give it one.
    UnifiedVector<T*> pv(static_cast<size_t>(batch));
    UnifiedVector<T*> pv0(static_cast<size_t>(batch));
    auto Aview  = sub<T>(A.data(),  0, n, 0, n, ld, st, batch, pv.data());
    auto A0view = sub<T>(A0.data(), 0, n, 0, n, ld, st, batch, pv0.data());
    auto reset = [&] { MatrixView<T, MatrixFormat::Dense>::copy(*q, Aview, A0view); q->wait(); };
    reset();
    size_t bytes = potrf_buffer_size(*q, Aview, Uplo::Lower);
    UnifiedVector<std::byte> ws(std::max<size_t>(bytes, 1));
    auto call = [&] { potrf(*q, Aview, Uplo::Lower, ws.to_span(), info.to_span()); q->wait(); };
    auto w0 = std::chrono::steady_clock::now();
    do { call(); reset(); }
    while (std::chrono::duration<double>(std::chrono::steady_clock::now()-w0).count() < warm_s);
    std::vector<double> ms;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        call();
        auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1-t0).count());
        if (r + 1 < reps) reset();
    }
    Stats s = summarize(ms);
    const double res = std::max(residual<T>(A.data(), A0.data(), n, ld, st, 0),
                                residual<T>(A.data(), A0.data(), n, ld, st, batch - 1));
    const double fs = (std::string(tn).front() == 'c') ? 4.0 : 1.0;
    const double gf = fs * (double(n)*double(n)*double(n)/3.0) * double(batch) / (s.med * 1e6);
    int badv = 0; for (int i = 0; i < batch; ++i) if (info[size_t(i)] != 0) ++badv;
    std::printf("vendorpotrf,%s,%d,%d,%.6f,%.6f,%.4f,%.2f,%.3e,%d\n",
                tn, n, batch, s.med, s.mn, s.rel_sd, gf, res, badv);
    std::fflush(stdout);
    return 0;
}


// ------------------------------------------- the panel solve, differentially
// The blocked driver returns a wrong answer whenever the panel trsm takes the
// NATIVE route. This mode takes potrf out of the picture: one well-conditioned
// L11 and one panel B, solved twice from the same input -- once through
// sycl_trsm::trsm_native_blocked and once through backend::trsm_vendor -- and
// compared against each other AND against a host forward-substitution residual,
// so neither is assumed correct.
template <typename T>
static int mode_trsmdiff(const char* tn, int n, int ib, int j, int batch,
                         bool flat, int slack, int rep) {
    using R = typename RealOf<T>::type;
    auto q = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);
    const int ld = n, st = n * n;
    const int m2 = n - j - ib;
    if (m2 <= 0) { std::fprintf(stderr, "m2 <= 0\n"); return 2; }

    // `flat`: A is ib x ib at ld == ib and B is m2 x ib at ld == m2, freshly
    // allocated. `sub`: both are sub-views of one n x n parent. Same shapes,
    // same arithmetic; the ONLY difference is the leading dimension and the
    // batch stride the views carry.
    // `slack`: extra elements past the parent, because the 6-arg MatrixView
    // constructor sizes data_ as stride*batch FROM THE OFFSET POINTER
    // (matrix.cc:1839), so a sub-view's nominal span runs off the end of the
    // parent allocation. Slack makes that span legal without changing any
    // ld/stride, which separates "over-long span" from "wrong ld".
    const int Ald = flat ? ib : ld;
    const int Ast = flat ? ib * ib : st;
    const int Bld = flat ? m2 : ld;
    const int Bst = flat ? m2 * ib : st;
    UnifiedVector<T> P(flat ? size_t(1)
                            : size_t(st) * size_t(batch) + size_t(slack));
    UnifiedVector<T> FA(flat ? size_t(Ast) * size_t(batch) + size_t(slack) : size_t(1));
    UnifiedVector<T> FB(flat ? size_t(Bst) * size_t(batch) + size_t(slack) : size_t(1));
    T* Abase = flat ? FA.data() : P.data();
    T* Bbase = flat ? FB.data() : P.data();
    const int Ar0 = flat ? 0 : j,  Ac0 = flat ? 0 : j;
    const int Br0 = flat ? 0 : j + ib, Bc0 = flat ? 0 : j;
    UnifiedVector<T> Bref(size_t(m2) * size_t(ib) * size_t(batch));
    UnifiedVector<T> Xn(size_t(m2) * size_t(ib) * size_t(batch));
    Rng r;
    for (size_t i = 0; i < P.size(); ++i) P.data()[i] = from_rng<T>(r);
    for (size_t i = 0; i < FA.size(); ++i) FA.data()[i] = from_rng<T>(r);
    for (size_t i = 0; i < FB.size(); ++i) FB.data()[i] = from_rng<T>(r);
    // L11: unit-ish diagonal, small off-diagonal -> condition number ~1.
    for (int b = 0; b < batch; ++b) {
        T* A = Abase + size_t(b) * size_t(Ast);
        for (int c = 0; c < ib; ++c)
            for (int i = c; i < ib; ++i)
                A[size_t(Ac0 + c) * size_t(Ald) + size_t(Ar0 + i)] =
                    (i == c) ? T(double(ib)) : from_rng<T>(r);
    }
    for (size_t i = 0; i < Bref.size(); ++i) Bref.data()[i] = from_rng<T>(r);

    auto load_B = [&] {
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < ib; ++c)
                for (int i = 0; i < m2; ++i)
                    Bbase[size_t(b)*size_t(Bst) + size_t(Bc0+c)*size_t(Bld) + size_t(Br0+i)]
                        = Bref.data()[size_t(b)*size_t(m2)*size_t(ib) + size_t(c)*size_t(m2) + size_t(i)];
    };
    auto save_B = [&](UnifiedVector<T>& out) {
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < ib; ++c)
                for (int i = 0; i < m2; ++i)
                    out.data()[size_t(b)*size_t(m2)*size_t(ib) + size_t(c)*size_t(m2) + size_t(i)]
                        = Bbase[size_t(b)*size_t(Bst) + size_t(Bc0+c)*size_t(Bld) + size_t(Br0+i)];
    };

    UnifiedVector<T*> pA(static_cast<size_t>(batch));
    UnifiedVector<T*> pB(static_cast<size_t>(batch));
    auto A11 = sub<T>(Abase, Ar0, ib, Ac0, ib, Ald, Ast, batch, pA.data());
    auto A21 = sub<T>(Bbase, Br0, m2, Bc0, ib, Bld, Bst, batch, pB.data());

    load_B();
    sycl_trsm::trsm_native_blocked<T>(*q, A11, A21, T(1), Side::Right, Uplo::Lower,
                                      Transpose::ConjTrans, Diag::NonUnit);
    q->wait();
    save_B(Xn);

    load_B();
    backend::trsm_vendor<Backend::CUDA, T>(*q, A11, A21, Side::Right, Uplo::Lower,
                                           Transpose::ConjTrans, Diag::NonUnit, T(1));
    q->wait();
    // P now holds the vendor answer in the A21 slot.

    // X op(A) = B with op(A) = L11^H, i.e. X(:,c) solved forward in c.
    auto resid = [&](const T* X, int b) {
        const T* A = Abase + size_t(b) * size_t(Ast);
        R num = 0, den = 0;
        for (int i = 0; i < m2; ++i)
            for (int c = 0; c < ib; ++c) {
                T acc = T(0);
                for (int p2 = 0; p2 <= c; ++p2)
                    acc += X[size_t(b)*size_t(m2)*size_t(ib) + size_t(p2)*size_t(m2) + size_t(i)]
                         * conj_of(A[size_t(Ac0 + p2)*size_t(Ald) + size_t(Ar0 + c)]);
                const T d = acc - Bref.data()[size_t(b)*size_t(m2)*size_t(ib) + size_t(c)*size_t(m2) + size_t(i)];
                num = std::max(num, R(std::abs(d)));
                den = std::max(den, R(std::abs(Bref.data()[size_t(b)*size_t(m2)*size_t(ib) + size_t(c)*size_t(m2) + size_t(i)])));
            }
        return double(num) / double(den);
    };

    // A11 is unchanged by either call, so the residual above may use P's copy.
    UnifiedVector<T> Xv(size_t(m2) * size_t(ib) * size_t(batch));
    save_B(Xv);

    double maxrel = 0.0;
    int items_diff = 0;
    for (int b = 0; b < batch; ++b) {
        R num = 0, den = 0;
        for (size_t i = 0; i < size_t(m2) * size_t(ib); ++i) {
            const size_t k = size_t(b) * size_t(m2) * size_t(ib) + i;
            num = std::max(num, R(std::abs(Xn.data()[k] - Xv.data()[k])));
            den = std::max(den, R(std::abs(Xv.data()[k])));
        }
        const double rel = double(num) / double(den);
        if (rel > 1e-3) ++items_diff;
        maxrel = std::max(maxrel, rel);
    }
    const double rn0 = resid(Xn.data(), 0), rnL = resid(Xn.data(), batch - 1);
    const double rv0 = resid(Xv.data(), 0), rvL = resid(Xv.data(), batch - 1);
    std::printf("trsmdiff,%s,%d,%d,%d,%d,%d,%s,%d,%d,%.3e,%d,%.3e,%.3e,%.3e,%.3e\n",
                tn, n, ib, j, m2, batch, flat ? "flat" : "sub", slack, rep,
                maxrel, items_diff, rn0, rnL, rv0, rvL);
    std::fflush(stdout);
    return 0;
}

template <typename T>
static int dispatch_mode(const char* tn, int argc, char** argv) {
    const std::string mode = argv[1];
    auto I = [&](int i) { return std::atoi(argv[i]); };
    if (mode == "panel")       return mode_panel<T>(tn, I(3), I(4), I(5), I(6), false, 0, 0);
    if (mode == "panelflat")   return mode_panel<T>(tn, 0, 0, I(5), I(6), true, I(3), I(4));
    if (mode == "trail")       return mode_trail<T>(tn, I(3), I(4), I(5), I(6),
                                                    std::string(argv[7]) == "flat",
                                                    std::string(argv[8]) == "T" ? Transpose::Trans
                                                                                : Transpose::ConjTrans,
                                                    I(9));
    if (mode == "blocked")     return mode_blocked<T>(tn, I(3), I(4), I(5), I(6), I(7));
    if (mode == "vendorpotrf") return mode_vendorpotrf<T>(tn, I(3), I(4), I(5));
    if (mode == "trsmdiff")    return mode_trsmdiff<T>(tn, I(3), I(4), I(5), I(6),
                                                        std::string(argv[7]) == "flat", I(8), I(9));
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}

int main(int argc, char** argv) {
    g_break = std::getenv("PHASE2_BREAK");
    if (g_break && !*g_break) g_break = nullptr;
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: phase2 <mode> <type> ...\n"
            "  panel       <type> <n> <nb> <batch> <reps>\n"
            "  panelflat   <type> <m2> <ib> <batch> <reps>\n"
            "  trail       <type> <m> <n> <k> <batch> <sub|flat> <T|C> <reps>\n"
            "  blocked     <type> <n> <nb> <W> <batch> <reps>\n"
            "  vendorpotrf <type> <n> <batch> <reps>\n");
        return 2;
    }
    const std::string t = argv[2];
    if (t == "float")   return dispatch_mode<float>("float", argc, argv);
    if (t == "double")  return dispatch_mode<double>("double", argc, argv);
    if (t == "cfloat")  return dispatch_mode<std::complex<float>>("cfloat", argc, argv);
    if (t == "cdouble") return dispatch_mode<std::complex<double>>("cdouble", argc, argv);
    std::fprintf(stderr, "unknown type %s\n", t.c_str());
    return 2;
}
