// WP6 QUESTION 3: what does partial pivoting cost a CTA-resident batched LU?
//
// The plan says: "the risk is that [the pivot search and the row swap] serialise
// the whole factorization at small n. Measure the un-pivoted variant as a lower
// bound to know how much the pivoting is costing." This is that probe, and it
// splits the cost into its two halves rather than reporting one number:
//
//   nopiv     Doolittle, no search, no swap                  -- the LOWER BOUND
//   swaponly  no search; swaps with a PRECOMPUTED pivot row  -- the swap alone
//   pivman    work-group argmax by an explicit SLM tree      -- search + swap
//   pivgrp    the same, with TWO sycl::reduce_over_group per column
//
// pivgrp exists for one reason: WP4 recorded that a reduce_over_group is what
// REOPENS the 48 KB launch hole, and WP5 walked into it anyway. Here the same
// kernel is written both ways at the same SLM footprint, so the hole can be
// attributed to the reduction rather than to the tile.
//
// STANDALONE SYCL. It links no BatchLAS library on purpose: there is no native
// getrf to call, and a probe that pulled in the dispatch layer would be timing
// code that does not exist yet.
//
// ONE (variant, type, n, batch) PER PROCESS, deliberately. The SLM attribute is
// STICKY PER CUfunction: any earlier, larger launch in the same process raises
// the cap and hides the hole by execution order. A ladder run inside one process
// is not a ladder, it is one measurement repeated.
#include <sycl/sycl.hpp>

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

// ------------------------------------------------------------------ scalars
template <class T> struct RealOf { using type = T; };
template <class R> struct RealOf<std::complex<R>> { using type = R; };
template <class T> struct Prom { using type = double; };
template <class R> struct Prom<std::complex<R>> { using type = std::complex<double>; };

static inline double ab(double x) { return std::fabs(x); }
static inline double ab(std::complex<double> x) { return std::abs(x); }
static inline double up(float x) { return double(x); }
static inline double up(double x) { return x; }
static inline std::complex<double> up(std::complex<float> x) { return {double(x.real()), double(x.imag())}; }
static inline std::complex<double> up(std::complex<double> x) { return x; }

template <class T> static inline T mk(double re, double im);
template <> inline float mk<float>(double re, double) { return float(re); }
template <> inline double mk<double>(double re, double) { return re; }
template <> inline std::complex<float> mk<std::complex<float>>(double re, double im) { return {float(re), float(im)}; }
template <> inline std::complex<double> mk<std::complex<double>>(double re, double im) { return {re, im}; }

// |z| WITHOUT std::abs on a complex: LAPACK's own getrf pivots on
// |Re| + |Im| (cabs1), which is cheaper and is what the vendor does. Using the
// true modulus here would change WHICH row is selected on some matrices, so the
// probe would no longer be measuring the same algorithm.
template <class T> static inline auto cabs1(const T& x) { return sycl::fabs(x); }
template <class R> static inline R cabs1(const std::complex<R>& x) {
    return sycl::fabs(x.real()) + sycl::fabs(x.imag());
}

// HAND-ROLLED complex multiply and reciprocal. std::complex's operator* emits an
// isnan branch and a call to __mulsc3 in device code, and operator/ emits
// __divsc3 -- the Annex G trap, worth 1.2-1.3x in a hot loop. The rank-1 update
// below is the hot loop, so it must not contain either.
template <class T> static inline T cmul(const T& a, const T& b) { return a * b; }
template <class R> static inline std::complex<R> cmul(const std::complex<R>& a,
                                                      const std::complex<R>& b) {
    return {a.real() * b.real() - a.imag() * b.imag(),
            a.real() * b.imag() + a.imag() * b.real()};
}
template <class T> static inline T crecip(const T& a) { return T(1) / a; }
template <class R> static inline std::complex<R> crecip(const std::complex<R>& a) {
    const R d = a.real() * a.real() + a.imag() * a.imag();
    return {a.real() / d, -a.imag() / d};
}

static inline double nanmax(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::quiet_NaN();
    return a > b ? a : b;
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(int32_t(uint32_t(s >> 32))) / 2147483648.0;
    }
};

struct Stat { double med, mean, relsd; };
static Stat stat_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const double med = v[v.size() / 2];
    double mean = 0; for (double x : v) mean += x; mean /= double(v.size());
    double sd = 0; for (double x : v) sd += (x - mean) * (x - mean);
    sd = std::sqrt(sd / double(v.size()));
    return {med, mean, mean > 0 ? sd / mean : 0.0};
}

enum class Var { NoPiv, SwapOnly, PivMan, PivGrp };

// ------------------------------------------------------------------ kernel
// One work-group per matrix; the whole matrix resident in local memory at
// ld = n | 1.
//
// The odd ld is potrf_cta.cc:555's choice, not geqrf_cta.cc's (which pads not at
// all). LU needs it: the ROW SWAP walks a row, i.e. `wg` work-items at stride
// ld, and an even ld puts every one of them in the same bank. geqrf has no such
// access pattern, which is why its header can say a pad "would only cost
// capacity".
template <typename T, Var V>
static sycl::event lu_launch(sycl::queue& q, T* A, const int* fixed_piv, int* piv_out,
                             int n, int ld, int batch, int wg, size_t tile_elems,
                             size_t red_slots) {
    using R = typename RealOf<T>::type;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> tile(sycl::range<1>(tile_elems), h);
        sycl::local_accessor<R, 1> rval(sycl::range<1>(red_slots), h);
        sycl::local_accessor<int, 1> ridx(sycl::range<1>(red_slots), h);
        sycl::local_accessor<int, 1> psel(sycl::range<1>(1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(batch) * size_t(wg)),
                                         sycl::range<1>(size_t(wg))),
                       [=](sycl::nd_item<1> it) {
            const int tid = int(it.get_local_id(0));
            const size_t b = it.get_group(0);
            auto g = it.get_group();
            T* S = &tile[0];
            const size_t st = size_t(n) * size_t(n);
            T* G = A + b * st;
            int* P = piv_out + b * size_t(n);
            const int* FP = fixed_piv + b * size_t(n);

            for (int e = tid; e < n * n; e += wg) {
                const int i = e % n, j = e / n;
                S[size_t(i) + size_t(j) * ld] = G[size_t(i) + size_t(j) * n];
            }
            it.barrier(sycl::access::fence_space::local_space);

            for (int k = 0; k < n; ++k) {
                int p = k;
                if constexpr (V == Var::PivMan) {
                    R bv = R(-1); int bi = k;
                    for (int i = k + tid; i < n; i += wg) {
                        const R v = cabs1(S[size_t(i) + size_t(k) * ld]);
                        if (v > bv) { bv = v; bi = i; }
                    }
                    rval[tid] = bv; ridx[tid] = bi;
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int s = wg / 2; s > 0; s >>= 1) {
                        if (tid < s && rval[tid + s] > rval[tid]) {
                            rval[tid] = rval[tid + s]; ridx[tid] = ridx[tid + s];
                        }
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    p = ridx[0];
                } else if constexpr (V == Var::PivGrp) {
                    R bv = R(-1); int bi = k;
                    for (int i = k + tid; i < n; i += wg) {
                        const R v = cabs1(S[size_t(i) + size_t(k) * ld]);
                        if (v > bv) { bv = v; bi = i; }
                    }
                    // TWO group reductions per column: one for the value, one to
                    // pick the lowest index attaining it. This is the shape WP4
                    // named as the thing that reopens the 48 KB hole.
                    const R m = sycl::reduce_over_group(g, bv, sycl::maximum<R>());
                    const int sel = sycl::reduce_over_group(g, (bv == m) ? bi : n,
                                                            sycl::minimum<int>());
                    if (tid == 0) psel[0] = sel;
                    it.barrier(sycl::access::fence_space::local_space);
                    p = psel[0];
                } else if constexpr (V == Var::SwapOnly) {
                    p = FP[k];
                }
                if constexpr (V != Var::NoPiv) {
                    if (tid == 0) P[k] = p + 1;      // 1-BASED, LAPACK ipiv
                    if (p != k) {
                        for (int j = tid; j < n; j += wg) {
                            const size_t a0 = size_t(k) + size_t(j) * ld;
                            const size_t a1 = size_t(p) + size_t(j) * ld;
                            const T t = S[a0]; S[a0] = S[a1]; S[a1] = t;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                } else {
                    if (tid == 0) P[k] = k + 1;
                }

                const T d = S[size_t(k) + size_t(k) * ld];
                const T r = crecip(d);
                for (int i = k + 1 + tid; i < n; i += wg)
                    S[size_t(i) + size_t(k) * ld] = cmul(S[size_t(i) + size_t(k) * ld], r);
                it.barrier(sycl::access::fence_space::local_space);

                const int m = n - k - 1;
                for (int e = tid; e < m * m; e += wg) {
                    const int i = k + 1 + (e % m), j = k + 1 + (e / m);
                    S[size_t(i) + size_t(j) * ld] =
                        S[size_t(i) + size_t(j) * ld]
                        - cmul(S[size_t(i) + size_t(k) * ld], S[size_t(k) + size_t(j) * ld]);
                }
                it.barrier(sycl::access::fence_space::local_space);
            }

            for (int e = tid; e < n * n; e += wg) {
                const int i = e % n, j = e / n;
                G[size_t(i) + size_t(j) * n] = S[size_t(i) + size_t(j) * ld];
            }
        });
    });
}

// || (P A0) x - L (U x) ||inf / || A0 x ||inf, on item 0 and the last item.
template <typename T>
static double probe(const std::vector<T>& F, const std::vector<T>& A0,
                    const std::vector<int>& piv, int n, int batch, int np) {
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
                for (int i = 0; i < n; ++i) ref[i] += up(A0[size_t(b) * st + size_t(j) * n + i]) * x[j];
            for (int k = 0; k < n; ++k) {
                const int ip = piv[size_t(b) * size_t(n) + size_t(k)] - 1;
                if (ip < 0 || ip >= n) return std::numeric_limits<double>::quiet_NaN();
                if (ip != k) std::swap(ref[k], ref[ip]);
            }
            std::vector<D> y(n, D(0)), z(n, D(0));
            for (int i = 0; i < n; ++i) {
                D acc = D(0);
                for (int j = i; j < n; ++j) acc += up(F[size_t(b) * st + size_t(j) * n + i]) * x[j];
                y[i] = acc;
            }
            for (int i = 0; i < n; ++i) {
                D acc = y[i];
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

static double warm_s() { const char* e = std::getenv("WARM_S"); return e ? std::atof(e) : 1.0; }

template <typename T, Var V>
static int run(const char* vn, const char* tn, int n, int batch, int reps) {
    using R = typename RealOf<T>::type;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};

    const size_t local_mem = q.get_device().get_info<sycl::info::device::local_mem_size>();
    const size_t budget = local_mem > 4096 ? local_mem - 4096 : 0;
    const int max_wg = int(q.get_device().get_info<sycl::info::device::max_work_group_size>());

    const int ld = n | 1;
    // WG overridable so the pivot delta can be reported against the work-group
    // width rather than at one arbitrary one: a wider group makes the trailing
    // update cheaper and the argmax tree deeper, and those pull opposite ways.
    int wg = 32; while (wg < n && wg * 2 <= max_wg && wg < 256) wg *= 2;
    if (const char* we = std::getenv("WG")) wg = std::atoi(we);
    if (wg < 32) wg = 32;
    if (wg > max_wg) wg = max_wg;
    const size_t red_slots = (V == Var::PivMan) ? size_t(wg) : 1;
    size_t tile_elems = size_t(ld) * size_t(n);
    // PAD=<bytes>: request EXACTLY this many bytes of local memory in total,
    // by growing the tile accessor past what the matrix needs.
    //
    // The n ladder alone cannot settle the 48 KB launch hole, because the hole
    // WP4 recorded is not a range but specific byte counts -- 48896 passes,
    // 49152 FAILS, 49664 passes -- and an n ladder steps over 49152 rather than
    // landing on it. This is the discriminating knob: same kernel, same shape,
    // one byte count at a time.
    const size_t base = tile_elems * sizeof(T) + red_slots * (sizeof(R) + sizeof(int)) + sizeof(int);
    if (const char* pe = std::getenv("PAD")) {
        const size_t want = size_t(std::atoll(pe));
        if (want > base) tile_elems += (want - base + sizeof(T) - 1) / sizeof(T);
    }
    const size_t slm = tile_elems * sizeof(T) + red_slots * (sizeof(R) + sizeof(int)) + sizeof(int);

    const size_t st = size_t(n) * n;
    std::vector<T> hA0(st * batch);
    std::vector<int> hFP(size_t(n) * batch), hP(size_t(n) * batch, 0);
    {
        Rng rg(12345);
        for (size_t i = 0; i < hA0.size(); ++i) hA0[i] = mk<T>(rg.next(), rg.next());
        // Diagonally dominant, then ROW-PERMUTED -- see the note in lubench.cpp:
        // on the dominant matrix alone partial pivoting picks the diagonal every
        // time, so the pivot path is never exercised and nopiv/pivman would be
        // measuring the same swap count (zero).
        std::vector<T> col(n);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i)
                hA0[size_t(b) * st + size_t(i) * n + i] =
                    hA0[size_t(b) * st + size_t(i) * n + i] + mk<T>(double(n), 0.0);
            std::vector<int> perm(n);
            for (int i = 0; i < n; ++i) perm[i] = i;
            for (int i = n - 1; i > 0; --i) {
                const int j = int((rg.next() * 0.5 + 0.5) * double(i + 1)) % (i + 1);
                std::swap(perm[i], perm[j]);
            }
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) col[i] = hA0[size_t(b) * st + size_t(j) * n + size_t(perm[i])];
                for (int i = 0; i < n; ++i) hA0[size_t(b) * st + size_t(j) * n + i] = col[i];
            }
        }
    }

    T* dA = sycl::malloc_device<T>(st * batch, q);
    int* dFP = sycl::malloc_device<int>(size_t(n) * batch, q);
    int* dP = sycl::malloc_device<int>(size_t(n) * batch, q);
    if (!dA || !dFP || !dP) { std::printf("%s,%s,%d,%d,ALLOC_FAIL\n", vn, tn, n, batch); return 1; }

    // SwapOnly needs a pivot list that is not the identity but costs no search.
    // Take it from a PivMan run so the swap COUNT is the real one -- a synthetic
    // list with a different number of swaps would not be the same amount of work.
    {
        std::vector<int> fp(size_t(n) * batch);
        Rng rg(31337);
        for (int b = 0; b < batch; ++b)
            for (int k = 0; k < n; ++k)
                fp[size_t(b) * size_t(n) + size_t(k)] =
                    k + int((rg.next() * 0.5 + 0.5) * double(n - k)) % (n - k);
        q.memcpy(dFP, fp.data(), fp.size() * sizeof(int)).wait();
    }

    auto once = [&] {
        q.memcpy(dA, hA0.data(), hA0.size() * sizeof(T)).wait();
        lu_launch<T, V>(q, dA, dFP, dP, n, ld, batch, wg, tile_elems, red_slots);
        q.wait();
    };

    try {
        const auto w0 = std::chrono::steady_clock::now();
        do { once(); }
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s());
    } catch (const sycl::exception& e) {
        // THE LAUNCH HOLE SHOWS UP HERE, as a failed submit, and this row is the
        // point of the ladder: it must be reported, not swallowed.
        std::printf("%s,%s,%d,%d,%d,%d,%zu,-1,-1,-1,LAUNCH_FAIL:%s\n",
                    vn, tn, n, batch, wg, ld, slm, e.what());
        sycl::free(dA, q); sycl::free(dFP, q); sycl::free(dP, q);
        return 0;
    }

    std::vector<double> ms;
    for (int r = 0; r < reps; ++r) {
        q.memcpy(dA, hA0.data(), hA0.size() * sizeof(T)).wait();
        const auto t0 = std::chrono::steady_clock::now();
        lu_launch<T, V>(q, dA, dFP, dP, n, ld, batch, wg, tile_elems, red_slots);
        q.wait();
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    const Stat s = stat_of(ms);

    std::vector<T> hF(st * batch);
    q.memcpy(hF.data(), dA, hF.size() * sizeof(T)).wait();
    q.memcpy(hP.data(), dP, hP.size() * sizeof(int)).wait();
    const double res = probe<T>(hF, hA0, hP, n, batch, 2);
    int ntp = 0; for (int k = 0; k < n; ++k) if (hP[k] != k + 1) ++ntp;

    const double dn = double(n);
    const double gf = double(batch) * (2.0 / 3.0) * dn * dn * dn;
    std::printf("%s,%s,%d,%d,%d,%d,%zu,%.4f,%.4f,%.4f,%.2f,%.3e,%d,%s\n",
                vn, tn, n, batch, wg, ld, slm, s.med, s.mean, s.relsd,
                gf / (s.med * 1e6), res, ntp,
                std::isfinite(res) ? "ok" : "BAD");

    (void)budget;
    sycl::free(dA, q); sycl::free(dFP, q); sycl::free(dP, q);
    return 0;
}

template <typename T>
static int dispatch_var(const std::string& v, const char* tn, int n, int b, int r) {
    if (v == "nopiv")    return run<T, Var::NoPiv>("nopiv", tn, n, b, r);
    if (v == "swaponly") return run<T, Var::SwapOnly>("swaponly", tn, n, b, r);
    if (v == "pivman")   return run<T, Var::PivMan>("pivman", tn, n, b, r);
    if (v == "pivgrp")   return run<T, Var::PivGrp>("pivgrp", tn, n, b, r);
    std::fprintf(stderr, "unknown variant %s\n", v.c_str());
    return 2;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
            "usage: pivotcost <variant> <type> <n> <batch> <reps>\n"
            "variants: nopiv | swaponly | pivman | pivgrp\n"
            "types   : float double cfloat cdouble\n"
            "cols    : variant,type,n,batch,wg,ld,slm_bytes,med_ms,mean_ms,relsd,GFLOPs,resid,ntpiv,flag\n");
        return 2;
    }
    const std::string v = argv[1], t = argv[2];
    const int n = std::atoi(argv[3]), b = std::atoi(argv[4]), r = std::atoi(argv[5]);
    try {
        if (t == "float")   return dispatch_var<float>(v, "float", n, b, r);
        if (t == "double")  return dispatch_var<double>(v, "double", n, b, r);
        if (t == "cfloat")  return dispatch_var<std::complex<float>>(v, "cfloat", n, b, r);
        if (t == "cdouble") return dispatch_var<std::complex<double>>(v, "cdouble", n, b, r);
    } catch (const std::exception& e) {
        std::printf("%s,%s,%d,%d,THREW,%s\n", v.c_str(), t.c_str(), n, b, e.what());
        return 0;
    }
    std::fprintf(stderr, "unknown type %s\n", t.c_str());
    return 2;
}
