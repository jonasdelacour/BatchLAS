// WP4 Phase 2 -- correctness proof for the blocked native potrf driver.
//
// Modes (all take <type> <n> <batch>; type in f|d|c|z):
//   facade   -- batchlas::potrf<CUDA,T> through the facade, honouring
//               BATCHLAS_POTRF_ROUTE. Checks: host multiply-back residual on
//               items 0 AND batch-1 (item 0 sits at offset 0 and cannot move
//               when the batch stride is wrong -- a recorded blind guard),
//               bit-exact survival of the POISONED UPPER TRIANGLE, and info==0.
//   direct   -- the same, but calling sycl_potrf::potrf_blocked_dispatch
//               directly. A direct call cannot be served by a vendor.
//   bitexact -- runs both on identical copies and prints the max |difference|.
//               Zero is the proof that the facade reached the driver rather
//               than cuSOLVER. Run it with BATCHLAS_GEMM_ROUTE=native and
//               BATCHLAS_TRSM_ROUTE=native so the injected seams resolve to the
//               same kernels the direct call defaults to.
//   info     -- plants non-PD items and checks the GLOBAL 1-based index and
//               first-failure-wins, plus finiteness of a failed item's A.
//   oq9      -- open question 9: one panel of the schedule, by hand, reporting
//               max |imag(diag(A22))| before and after.
#include <batchlas/blas/linalg.hh>
#include <batchlas/backend_config.h>
#include "../../../src/extensions/potrf_native.hh"
#include "../../../src/extensions/symmetric_product_fold.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <batchlas/util/mempool.hh>
#include <limits>
#include <vector>

using namespace batchlas;

template <typename T> struct RealOf { using type = T; };
template <typename U> struct RealOf<std::complex<U>> { using type = U; };
template <typename T> struct IsCplx : std::false_type {};
template <typename U> struct IsCplx<std::complex<U>> : std::true_type {};

static float  conj_of(float v) { return v; }
static double conj_of(double v) { return v; }
static std::complex<float>  conj_of(std::complex<float> v)  { return std::conj(v); }
static std::complex<double> conj_of(std::complex<double> v) { return std::conj(v); }
static double imag_of(float) { return 0.0; }
static double imag_of(double) { return 0.0; }
static double imag_of(std::complex<float> v) { return double(v.imag()); }
static double imag_of(std::complex<double> v) { return double(v.imag()); }
static double real_of(float v) { return v; }
static double real_of(double v) { return v; }
static double real_of(std::complex<float> v) { return double(v.real()); }
static double real_of(std::complex<double> v) { return double(v.real()); }
template <typename T> static bool finite_of(T v) {
    if constexpr (IsCplx<T>::value) return std::isfinite(v.real()) && std::isfinite(v.imag());
    else return std::isfinite(v);
}

struct Rng {
    uint32_t s = 0x9E3779B9u;
    double next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5;
                    return double(s) / 4294967296.0 - 0.5; }
};
template <typename T> static T from_rng(Rng& r);
template <> float from_rng<float>(Rng& r) { return float(r.next()); }
template <> double from_rng<double>(Rng& r) { return r.next(); }
template <> std::complex<float> from_rng<std::complex<float>>(Rng& r) {
    return {float(r.next()), float(r.next())};
}
template <> std::complex<double> from_rng<std::complex<double>>(Rng& r) {
    return {r.next(), r.next()};
}

// Fill the WHOLE allocation with junk, then write an HPD matrix into the LOWER
// triangle of the named n x n window only. The upper triangle keeps its junk:
// that is the poison whose bit-exact survival the checks below assert, which is
// the one property a lower-triangle residual is structurally blind to (the
// harness's own PHASE2_BREAK=nofold stayed green on all four types).
template <typename T>
static void fill_hpd_lower_only(T* p, int n, int ld, int stride, int batch, size_t total) {
    Rng r;
    for (size_t i = 0; i < total; ++i) p[i] = from_rng<T>(r);

    std::vector<T> G(size_t(n) * size_t(n));
    {
        std::vector<T> M(size_t(n) * size_t(n));
        Rng q; q.s = 12345u;
        for (auto& v : M) v = from_rng<T>(q);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                T acc = T(0);
                for (int k = 0; k < n; ++k)
                    acc += conj_of(M[size_t(i)*size_t(n)+size_t(k)]) * M[size_t(j)*size_t(n)+size_t(k)];
                G[size_t(j)*size_t(n)+size_t(i)] = acc;
            }
        for (int i = 0; i < n; ++i) {
            T d = G[size_t(i)*size_t(n)+size_t(i)];
            G[size_t(i)*size_t(n)+size_t(i)] = T(real_of(d));   // exactly real diagonal
        }
    }
    for (int b = 0; b < batch; ++b) {
        T* A = p + size_t(b) * size_t(stride);
        const double shift = double(n) * (1.0 + 0.01 * double(b % 17));
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i) {   // LOWER TRIANGLE ONLY
                T v = G[size_t(j)*size_t(n)+size_t(i)];
                if (i == j) v = v + T(shift);
                A[size_t(j)*size_t(ld)+size_t(i)] = v;
            }
    }
}

// max_{i>=j} |(L L^H - A)_{ij}| / max |A_{ij}|, over the lower triangle.
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
                acc += Lb[size_t(t)*size_t(ld)+size_t(i)] * conj_of(Lb[size_t(t)*size_t(ld)+size_t(j)]);
            const T d = acc - Ab[size_t(j)*size_t(ld)+size_t(i)];
            num = std::max(num, R(std::abs(d)));
            den = std::max(den, R(std::abs(Ab[size_t(j)*size_t(ld)+size_t(i)])));
        }
    return double(num) / double(den);
}

// Every element OUTSIDE the lower triangle of the named window, bit for bit.
template <typename T>
static long upper_bits_changed(const T* now, const T* was, int n, int ld, int stride, int batch) {
    long bad = 0;
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < j; ++i) {
                const size_t k = size_t(b)*size_t(stride)+size_t(j)*size_t(ld)+size_t(i);
                if (std::memcmp(&now[k], &was[k], sizeof(T)) != 0) ++bad;
            }
    return bad;
}

template <typename T>
static MatrixView<T, MatrixFormat::Dense> sub(T* base, int r0, int nr, int c0, int nc,
                                              int ld, int stride, int batch, T** ptrs = nullptr) {
    return MatrixView<T, MatrixFormat::Dense>(
        base + std::ptrdiff_t(c0) * ld + r0, nr, nc, ld, stride, batch, ptrs);
}

static const char* g_mode = "facade";

template <typename T>
static int run(const char* tn, int n, int batch) {
    auto q = std::make_shared<Queue>(Device("gpu"), Backend::CUDA);

    // A NON-TRIVIAL ld AND a stride that is not ld*cols: every sub-view the
    // driver builds carries these, and the [FIX-B-trap] failure mode is
    // invisible at ld == rows.
    const int ld = n + 7;
    const int stride = ld * n + 13;
    const size_t total = size_t(stride) * size_t(batch);

    UnifiedVector<T> Abuf(total);
    UnifiedVector<T> Bbuf(total);
    UnifiedVector<int32_t> info(batch);
    fill_hpd_lower_only<T>(Abuf.data(), n, ld, stride, batch, total);
    std::vector<T> orig(Abuf.data(), Abuf.data() + total);

    auto A = sub<T>(Abuf.data(), 0, n, 0, n, ld, stride, batch);

    const int ceiling = sycl_potrf::potrf_cta_max_n<T>();

    if (std::strcmp(g_mode, "sizing") == 0) {
        // src/extensions/ortho.cc:72-78 verbatim in shape: a view over
        // MEASURING-MODE (unbacked) workspace addresses, handed to the public
        // potrf_buffer_size. Dereferencing one faults (mempool.hh:33-37), so
        // this is the check that the new max()-over-native-tiers query stayed
        // pure with respect to memory contents.
        auto sizer = BumpAllocator::measuring();
        auto ATA = sizer.allocate<T>(*q, size_t(n) * size_t(n) * size_t(batch));
        auto ptrs = sizer.allocate<T*>(*q, size_t(batch));
        auto C = MatrixView<T, MatrixFormat::Dense>(ATA.data(), n, n, n, n * n, batch, ptrs.data());
        const size_t bytes = potrf_buffer_size<Backend::CUDA, T>(*q, C, Uplo::Lower);
        std::printf("sizing,%s,n=%d,batch=%d,potrf_buffer_size=%zu\n", tn, n, batch, bytes);
        std::printf("VERDICT sizing %s: PASS\n", tn);
        return 0;
    }

    if (std::strcmp(g_mode, "info") == 0) {
        // Plant failures. A(c,c) := -1 makes the updated pivot at column c
        // strictly negative and leaves every earlier pivot untouched, so the
        // reported minor order is exactly c+1.
        const int c0 = n / 3, c1 = 2 * n / 3;
        for (int b = 0; b < batch; ++b) {
            T* Ab = Abuf.data() + size_t(b) * size_t(stride);
            if (b == 1) Ab[size_t(c0)*size_t(ld)+size_t(c0)] = T(-1);
            if (b == 2) { Ab[size_t(c1)*size_t(ld)+size_t(c1)] = T(-1);
                          Ab[size_t(c0)*size_t(ld)+size_t(c0)] = T(-1); }  // first must win
            if (b == batch-1) Ab[size_t(c1)*size_t(ld)+size_t(c1)] = T(-1);
            // A NaN pivot, not merely a negative one. The leaf's !(akk > 0)
            // catches both, but only the NaN makes the UNQUENCHED path produce
            // NaN downstream -- a negative pivot divides to a finite number.
            if (b == 3) Ab[size_t(c0)*size_t(ld)+size_t(c0)] =
                T(std::numeric_limits<typename RealOf<T>::type>::quiet_NaN());
        }
        for (int b = 0; b < batch; ++b) info[b] = -12345;   // caller garbage
        auto ws = UnifiedVector<std::byte>(
            sycl_potrf::potrf_blocked_buffer_size<T>(*q, A, Uplo::Lower));
        sycl_potrf::potrf_blocked_dispatch<T>(*q, A, Uplo::Lower, ws.to_span(), info.to_span());
        q->wait();
        long nonfinite = 0;
        for (int b : {1, 2, 3, batch-1}) {
            T* Ab = Abuf.data() + size_t(b) * size_t(stride);
            for (int j = 0; j < n; ++j)
                for (int i = j; i < n; ++i)
                    if (!finite_of(Ab[size_t(j)*size_t(ld)+size_t(i)])) ++nonfinite;
        }
        long healthy_bad = 0;
        double worst_healthy = 0;
        for (int b = 0; b < batch; ++b) {
            if (b == 1 || b == 2 || b == 3 || b == batch-1) continue;
            if (info[b] != 0) ++healthy_bad;
            worst_healthy = std::max(worst_healthy,
                                     residual<T>(Abuf.data(), orig.data(), n, ld, stride, b));
        }
        std::printf("info,%s,%d,%d,expect(%d,%d,%d,%d)=(%d,%d,%d,%d),healthy_info_bad=%ld,"
                    "healthy_res=%.3e,nonfinite=%ld\n",
                    tn, n, batch, c0+1, c0+1, c0+1, c1+1,
                    info[1], info[2], info[3], info[batch-1], healthy_bad, worst_healthy, nonfinite);
        const bool ok = info[1] == c0+1 && info[2] == c0+1 && info[3] == c0+1 &&
                        info[batch-1] == c1+1 &&
                        healthy_bad == 0 && nonfinite == 0 && worst_healthy < 1e-3;
        std::printf("VERDICT info %s: %s\n", tn, ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }

    if (std::strcmp(g_mode, "oq9") == 0) {
        if constexpr (!IsCplx<T>::value) {
            std::printf("oq9,%s,skipped (real scalar)\n", tn);
            return 0;
        } else {
            const int nb = std::min(ceiling, 96) / 32 * 32;
            const int ib = std::min(nb, n);
            const int m2 = n - ib;
            double imag_before = 0, real_before = 0;
            for (int b = 0; b < batch; ++b) {
                const T* Ab = Abuf.data() + size_t(b)*size_t(stride);
                for (int d = ib; d < n; ++d) {
                    const T v = Ab[size_t(d)*size_t(ld)+size_t(d)];
                    imag_before = std::max(imag_before, std::abs(imag_of(v)));
                    real_before = std::max(real_before, std::abs(real_of(v)));
                }
            }
            UnifiedVector<T*> p1(batch), p2(batch);
            auto A11 = sub<T>(Abuf.data(), 0, ib, 0, ib, ld, stride, batch, p1.data());
            sycl_potrf::potrf_cta_dispatch<T>(*q, A11, Uplo::Lower, Span<std::byte>(), info.to_span());
            auto A21 = sub<T>(Abuf.data(), ib, m2, 0, ib, ld, stride, batch, p2.data());
            trsm<Backend::CUDA, T>(*q, A11, A21, T(1), Side::Right, Uplo::Lower,
                                   Transpose::ConjTrans, Diag::NonUnit);
            const int W = 32;
            UnifiedVector<T> scratch(size_t(W)*size_t(W)*size_t(batch));
            for (int c = 0; c < m2; c += W) {
                const int w = std::min(W, m2 - c);
                auto Lrow = sub<T>(Abuf.data(), ib+c, w, 0, ib, ld, stride, batch);
                auto Cd   = sub<T>(Abuf.data(), ib+c, w, ib+c, w, ld, stride, batch);
                auto Sc   = sub<T>(scratch.data(), 0, w, 0, w, W, W*W, batch);
                gemm<Backend::CUDA, T>(*q, Lrow, Lrow, Sc, T(-1), T(0),
                                       Transpose::NoTrans, Transpose::ConjTrans);
                detail::fold_symmetric_product_into_triangle<T>(*q, Cd, Sc, T(1), Uplo::Lower);
                const int mr = m2 - c - w;
                if (mr > 0) {
                    auto Lr = sub<T>(Abuf.data(), ib+c+w, mr, 0, ib, ld, stride, batch);
                    auto Cr = sub<T>(Abuf.data(), ib+c+w, mr, ib+c, w, ld, stride, batch);
                    gemm<Backend::CUDA, T>(*q, Lr, Lrow, Cr, T(-1), T(1),
                                           Transpose::NoTrans, Transpose::ConjTrans);
                }
            }
            q->wait();
            double imag_after = 0, real_after = 0;
            for (int b = 0; b < batch; ++b) {
                const T* Ab = Abuf.data() + size_t(b)*size_t(stride);
                for (int d = ib; d < n; ++d) {
                    const T v = Ab[size_t(d)*size_t(ld)+size_t(d)];
                    imag_after = std::max(imag_after, std::abs(imag_of(v)));
                    real_after = std::max(real_after, std::abs(real_of(v)));
                }
            }
            std::printf("oq9,%s,n=%d,nb=%d,batch=%d,imag_before=%.3e,real_before=%.3e,"
                        "imag_after=%.3e,real_after=%.3e,ratio=%.3e\n",
                        tn, n, ib, batch, imag_before, real_before, imag_after, real_after,
                        real_after > 0 ? imag_after / real_after : 0.0);
            return 0;
        }
    }

#ifndef NOVENDOR
    if (std::strcmp(g_mode, "vendorcmp") == 0) {
        // The vendor batched potrf/trsm call A.data_ptrs(ctx) (cublas.cc:1220)
        // and a MatrixView built by the 6-arg constructor has none, so the
        // reference leg has to run over a Matrix, which owns one. ld == n and
        // stride == n*n here; the strided-ld coverage is the facade/direct modes.
        Matrix<T, MatrixFormat::Dense> Mv(n, n, batch), Mb(n, n, batch);
        const int vld = Mv.view().ld(), vst = Mv.view().stride();
        std::vector<T> src(size_t(vst) * size_t(batch));
        fill_hpd_lower_only<T>(src.data(), n, vld, vst, batch, src.size());
        std::copy(src.begin(), src.end(), Mv.view().data_ptr());
        std::copy(src.begin(), src.end(), Mb.view().data_ptr());
        UnifiedVector<int32_t> iv(batch), ib2(batch);
        {   // vendor
            auto V = Mv.view();
            auto ws = UnifiedVector<std::byte>(
                potrf_buffer_size<Backend::CUDA, T>(*q, V, Uplo::Lower));
            backend::potrf_vendor<Backend::CUDA, T>(*q, V, Uplo::Lower, ws.to_span(), iv.to_span());
            q->wait();
        }
        {   // blocked, direct (cannot be served by a vendor)
            auto V = Mb.view();
            auto ws = UnifiedVector<std::byte>(
                sycl_potrf::potrf_blocked_buffer_size<T>(*q, V, Uplo::Lower));
            sycl_potrf::potrf_blocked_dispatch<T>(*q, V, Uplo::Lower, ws.to_span(), ib2.to_span());
            q->wait();
        }
        double rel = 0, scale = 0;
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = j; i < n; ++i) {
                    const size_t k = size_t(b)*size_t(vst)+size_t(j)*size_t(vld)+size_t(i);
                    rel = std::max(rel, double(std::abs(Mv.view().data_ptr()[k] - Mb.view().data_ptr()[k])));
                    scale = std::max(scale, double(std::abs(Mv.view().data_ptr()[k])));
                }
        const double rv = residual<T>(Mv.view().data_ptr(), src.data(), n, vld, vst, batch-1);
        const double rb = residual<T>(Mb.view().data_ptr(), src.data(), n, vld, vst, batch-1);
        long ibad = 0; for (int b = 0; b < batch; ++b) if (iv[b] != 0 || ib2[b] != 0) ++ibad;
        const double tolv = (sizeof(typename RealOf<T>::type) == 4) ? 5e-5 : 1e-12;
        const bool okv = rb < tolv && rv < tolv && rel / (scale > 0 ? scale : 1) < tolv && ibad == 0;
        std::printf("vendorcmp,%s,n=%d,batch=%d,res_vendor=%.3e,res_blocked=%.3e,"
                    "max|L_v-L_b|/scale=%.3e,info_nonzero=%ld\n",
                    tn, n, batch, rv, rb, rel / (scale > 0 ? scale : 1), ibad);
        std::printf("VERDICT vendorcmp %s n=%d: %s\n", tn, n, okv ? "PASS" : "FAIL");
        return okv ? 0 : 1;
    }
#endif

    if (std::strcmp(g_mode, "bitexact") == 0) {
        std::copy(orig.begin(), orig.end(), Bbuf.data());
        auto Bv = sub<T>(Bbuf.data(), 0, n, 0, n, ld, stride, batch);
        UnifiedVector<int32_t> info2(batch);
        // facade, honouring BATCHLAS_POTRF_ROUTE
        auto wsA = UnifiedVector<std::byte>(potrf_buffer_size<Backend::CUDA, T>(*q, A, Uplo::Lower));
        potrf<Backend::CUDA, T>(*q, A, Uplo::Lower, wsA.to_span(), info.to_span());
        q->wait();
        // direct
        auto wsB = UnifiedVector<std::byte>(
            sycl_potrf::potrf_blocked_buffer_size<T>(*q, Bv, Uplo::Lower));
        sycl_potrf::potrf_blocked_dispatch<T>(*q, Bv, Uplo::Lower, wsB.to_span(), info2.to_span());
        q->wait();
        long diff_bits = 0; double maxdiff = 0;
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = j; i < n; ++i) {
                    const size_t k = size_t(b)*size_t(stride)+size_t(j)*size_t(ld)+size_t(i);
                    if (std::memcmp(&Abuf.data()[k], &Bbuf.data()[k], sizeof(T)) != 0) ++diff_bits;
                    maxdiff = std::max(maxdiff, double(std::abs(Abuf.data()[k] - Bbuf.data()[k])));
                }
        std::printf("bitexact,%s,%d,%d,diff_words=%ld,maxdiff=%.3e\n",
                    tn, n, batch, diff_bits, maxdiff);
        std::printf("VERDICT bitexact %s: %s\n", tn, diff_bits == 0 ? "PASS" : "FAIL");
        return diff_bits == 0 ? 0 : 1;
    }

    // facade / direct
    for (int b = 0; b < batch; ++b) info[b] = -12345;
    if (std::strcmp(g_mode, "direct") == 0) {
        auto ws = UnifiedVector<std::byte>(
            sycl_potrf::potrf_blocked_buffer_size<T>(*q, A, Uplo::Lower));
        sycl_potrf::potrf_blocked_dispatch<T>(*q, A, Uplo::Lower, ws.to_span(), info.to_span());
    } else {
        auto ws = UnifiedVector<std::byte>(potrf_buffer_size<Backend::CUDA, T>(*q, A, Uplo::Lower));
        potrf<Backend::CUDA, T>(*q, A, Uplo::Lower, ws.to_span(), info.to_span());
    }
    q->wait();

    const double r0 = residual<T>(Abuf.data(), orig.data(), n, ld, stride, 0);
    const double rN = residual<T>(Abuf.data(), orig.data(), n, ld, stride, batch - 1);
    const long upper = upper_bits_changed<T>(Abuf.data(), orig.data(), n, ld, stride, batch);
    long infobad = 0; for (int b = 0; b < batch; ++b) if (info[b] != 0) ++infobad;
    double imagdiag = 0;
    for (int b = 0; b < batch; ++b)
        for (int d = 0; d < n; ++d)
            imagdiag = std::max(imagdiag,
                std::abs(imag_of(Abuf.data()[size_t(b)*size_t(stride)+size_t(d)*size_t(ld)+size_t(d)])));

    const double tol = (sizeof(typename RealOf<T>::type) == 4) ? 5e-5 : 1e-12;
    const bool ok = r0 < tol && rN < tol && upper == 0 && infobad == 0 && imagdiag == 0.0;
    std::printf("%s,%s,n=%d,batch=%d,ceiling=%d,res0=%.3e,resN=%.3e,upper_changed=%ld,"
                "info_nonzero=%ld,max|imag(diag L)|=%.3e\n",
                g_mode, tn, n, batch, ceiling, r0, rN, upper, infobad, imagdiag);
    std::printf("VERDICT %s %s n=%d: %s\n", g_mode, tn, n, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: verify <mode> <type f|d|c|z> <n> <batch>\n");
        return 2;
    }
    g_mode = argv[1];
    const std::string t = argv[2];
    const int n = std::atoi(argv[3]);
    const int b = std::atoi(argv[4]);
    if (t == "f") return run<float>("float", n, b);
    if (t == "d") return run<double>("double", n, b);
    if (t == "c") return run<std::complex<float>>("cfloat", n, b);
    if (t == "z") return run<std::complex<double>>("cdouble", n, b);
    std::fprintf(stderr, "unknown type %s\n", t.c_str());
    return 2;
}
