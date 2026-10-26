// sygst blocked's diagonal block (sygst_native.hh). One work-item per column of a kb x kb block, the
// column in registers, the factor F and a transpose buffer in shared memory; 32 / kb items share a
// 32-wide work-group. Each pass is a triangular solve with F (itype 1) or a product with F^H (2, 3),
// applied to A and then to the conjugate transpose of the result, which is the Hermitian C itself.
// evidence: docs/perf/sygvd.md#sygst-the-blocked-family

#include <src/ops/sygst/sygst_native.hh>

#include <src/sycl/device_scalar.hh>
#include <src/sycl/kernel_attrs.hh>
#include <src/queue.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>

namespace batchlas::sycl_sygst {

namespace {

using sycl_device::Cx;

template <typename D> class SygstDiagKernel;

constexpr int kWg = kDiagMaxOrder;

// acc - a b, and acc + conj(a) b, written out (no std::complex in device code).
inline float sub_mul(float acc, float a, float b) { return sycl::fma(-a, b, acc); }
inline double sub_mul(double acc, double a, double b) { return sycl::fma(-a, b, acc); }
template <typename R>
inline Cx<R> sub_mul(Cx<R> acc, Cx<R> a, Cx<R> b) {
    acc.re = sycl::fma(-a.re, b.re, acc.re);
    acc.re = sycl::fma(a.im, b.im, acc.re);
    acc.im = sycl::fma(-a.re, b.im, acc.im);
    acc.im = sycl::fma(-a.im, b.re, acc.im);
    return acc;
}
inline float add_conj_mul(float acc, float a, float b) { return sycl::fma(a, b, acc); }
inline double add_conj_mul(double acc, double a, double b) { return sycl::fma(a, b, acc); }
template <typename R>
inline Cx<R> add_conj_mul(Cx<R> acc, Cx<R> a, Cx<R> b) {
    acc.re = sycl::fma(a.re, b.re, acc.re);
    acc.re = sycl::fma(a.im, b.im, acc.re);
    acc.im = sycl::fma(a.re, b.im, acc.im);
    acc.im = sycl::fma(-a.im, b.re, acc.im);
    return acc;
}
template <typename D>
inline D real_only(D x) {
    if constexpr (sycl_device::dev_is_complex_v<D>) return D{x.re, decltype(x.re)(0)};
    else return x;
}

// col := F^-1 col (forward substitution) or F^H col (top-down, in place), F lower at ld.
template <typename D>
inline void apply(D (&col)[kDiagMaxOrder], const D* f, int ld, int kb, bool inverse) {
    if (inverse) {
        BATCHLAS_UNROLL_FULL
        for (int i = 0; i < kDiagMaxOrder; ++i) {
            if (i < kb) {
                D s = col[i];
                BATCHLAS_UNROLL_FULL
                for (int p = 0; p < i; ++p) s = sub_mul(s, f[i + p * ld], col[p]);
                col[i] = sycl_device::dev_div_real(s, sycl_device::dev_real(f[i + i * ld]));
            }
        }
    } else {
        BATCHLAS_UNROLL_FULL
        for (int i = 0; i < kDiagMaxOrder; ++i) {
            if (i < kb) {
                D s = sycl_device::dev_mul_real(col[i], sycl_device::dev_real(f[i + i * ld]));
                BATCHLAS_UNROLL_FULL
                for (int p = i + 1; p < kDiagMaxOrder; ++p)
                    if (p < kb) s = add_conj_mul(s, f[p + i * ld], col[p]);
                col[i] = s;
            }
        }
    }
}

}  // namespace

template <typename T>
Event sygst_diag(Queue& ctx, T* a, int lda, std::ptrdiff_t stride_a, const T* b, int ldb, std::ptrdiff_t stride_b,
                 int kb, int batch, int itype, Uplo uplo) {
    using D = typename sycl_device::DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");
    if (kb < 1 || kb > kDiagMaxOrder || batch < 1) return ctx.get_event();
    D* const ap = reinterpret_cast<D*>(a);
    const D* const bp = reinterpret_cast<const D*>(b);
    const bool lower = uplo == Uplo::Lower, inverse = itype == 1;
    const int ipw = kWg / kb, ld = kb + 1, per = 2 * kb * ld, groups = (batch + ipw - 1) / ipw;
    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> slm(sycl::range<1>(static_cast<std::size_t>(ipw * per)), h);
        h.parallel_for<SygstDiagKernel<D>>(
            sycl::nd_range<1>(static_cast<std::size_t>(groups) * kWg, kWg), [=](sycl::nd_item<1> it) {
                const int t = static_cast<int>(it.get_local_id(0));
                const std::ptrdiff_t first = static_cast<std::ptrdiff_t>(it.get_group(0)) * ipw;
                D* const base = &slm[0];
                auto named = [=](int i, int c) { return lower ? i >= c : i <= c; };
                // The uplo triangles, coalesced: s(i, c) = A(i, c); F(i, c) = B(i, c) or conj(B(c, i)).
                for (int e = t; e < ipw * kb * kb; e += kWg) {
                    const int gi = e / (kb * kb), r = e - gi * kb * kb, i = r % kb, c = r / kb;
                    const std::ptrdiff_t itm = first + gi;
                    if (itm >= batch || !named(i, c)) continue;
                    D* const fg = base + gi * per;
                    fg[kb * ld + i + c * ld] = ap[itm * stride_a + static_cast<std::ptrdiff_t>(c) * lda + i];
                    const D bv = bp[itm * stride_b + static_cast<std::ptrdiff_t>(c) * ldb + i];
                    if (lower) fg[i + c * ld] = bv;
                    else fg[c + i * ld] = sycl_device::dev_conj(bv);
                }
                const int g = t / kb, j = t - g * kb;
                const bool active = g < ipw && first + g < batch;
                const D* const f = base + (active ? g * per : 0);
                D* const s = base + (active ? g * per : 0) + kb * ld;
                it.barrier(sycl::access::fence_space::local_space);
                D col[kDiagMaxOrder];
                BATCHLAS_UNROLL_FULL
                for (int i = 0; i < kDiagMaxOrder; ++i) {
                    col[i] = D{};
                    if (active && i < kb)
                        col[i] = i == j ? real_only(s[i + j * ld])
                                 : named(i, j) ? s[i + j * ld] : sycl_device::dev_conj(s[j + i * ld]);
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (active) apply(col, f, ld, kb, inverse);
                BATCHLAS_UNROLL_FULL
                for (int i = 0; i < kDiagMaxOrder; ++i)
                    if (active && i < kb) s[i + j * ld] = col[i];
                it.barrier(sycl::access::fence_space::local_space);
                // C = F^-1 X^H (or F^H X^H): X's conjugate transpose, row j of X as column j.
                BATCHLAS_UNROLL_FULL
                for (int i = 0; i < kDiagMaxOrder; ++i)
                    if (active && i < kb) col[i] = sycl_device::dev_conj(s[j + i * ld]);
                it.barrier(sycl::access::fence_space::local_space);
                if (active) apply(col, f, ld, kb, inverse);
                BATCHLAS_UNROLL_FULL
                for (int i = 0; i < kDiagMaxOrder; ++i)
                    if (active && i < kb) s[i + j * ld] = col[i];
                it.barrier(sycl::access::fence_space::local_space);
                for (int e = t; e < ipw * kb * kb; e += kWg) {
                    const int gi = e / (kb * kb), r = e - gi * kb * kb, i = r % kb, c = r / kb;
                    const std::ptrdiff_t itm = first + gi;
                    if (itm >= batch || !named(i, c)) continue;
                    const D v = base[gi * per + kb * ld + i + c * ld];
                    ap[itm * stride_a + static_cast<std::ptrdiff_t>(c) * lda + i] = i == c ? real_only(v) : v;
                }
            });
    });
    return ctx.get_event();
}

#define BATCHLAS_SYGST_DIAG_INSTANTIATE(T)                                                                  \
    template Event sygst_diag<T>(Queue&, T*, int, std::ptrdiff_t, const T*, int, std::ptrdiff_t, int, int, int, \
                                 Uplo);

BATCHLAS_SYGST_DIAG_INSTANTIATE(float)
BATCHLAS_SYGST_DIAG_INSTANTIATE(double)
BATCHLAS_SYGST_DIAG_INSTANTIATE(std::complex<float>)
BATCHLAS_SYGST_DIAG_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_SYGST_DIAG_INSTANTIATE

}  // namespace batchlas::sycl_sygst
