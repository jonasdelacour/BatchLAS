// sygst blocked's diagonal block (sygst_native.hh). One work-item per column of a kb x kb block, the
// column in registers; the factor F and a transpose buffer per item in shared memory, F stored so each
// pass reads its rows (or columns) as 16-byte packs. Each pass is a triangular solve with F (itype 1)
// or a product with F^H (2, 3), applied to A and then to the conjugate transpose of the result, which
// is the Hermitian C itself. Orders 32, 16 and 8 have compile-time instances.
// evidence: docs/perf/sygvd.md#sygst-the-blocked-family

#include <src/ops/sygst/sygst_native.hh>

#include <src/sycl/device_scalar.hh>
#include <src/sycl/kernel_attrs.hh>
#include <src/queue.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>

namespace batchlas::sycl_sygst {

namespace {

using sycl_device::Cx;

template <typename D, bool Inverse, int KB> class SygstDiagKernel;
template <typename D> class SygstMirrorKernel;

// acc - a b, acc + conj(a) b and a + b, written out (no std::complex in device code).
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
inline float add(float a, float b) { return a + b; }
inline double add(double a, double b) { return a + b; }
template <typename R>
inline Cx<R> add(Cx<R> a, Cx<R> b) { return Cx<R>{a.re + b.re, a.im + b.im}; }
template <typename D>
inline D real_only(D x) {
    if constexpr (sycl_device::dev_is_complex_v<D>) return D{x.re, decltype(x.re)(0)};
    else return x;
}

template <typename D>
struct Col {
    D v[kDiagMaxOrder];
};

template <typename D>
struct alignas(16) Pack {
    D e[16 / sizeof(D) > 0 ? 16 / sizeof(D) : 1];
};
template <typename D>
constexpr int kVec = static_cast<int>(sizeof(Pack<D>::e) / sizeof(D));

// F(r, q) in shared memory: row-major for the solve (it reads rows), column-major for the product.
template <bool Inverse>
inline int fidx(int r, int q, int ldf) { return Inverse ? r * ldf + q : r + q * ldf; }

// F^-1 c (forward substitution) or F^H c (top-down, in place). By value: an array out-parameter spills.
// Four partial sums per row shorten the dependent FMA chain.
template <bool Inverse, int KB, typename D>
inline Col<D> apply(Col<D> c, const D* f, int ldf, int kb) {
    constexpr int V = kVec<D>;
    BATCHLAS_UNROLL_FULL
    for (int i = 0; i < (KB ? KB : kDiagMaxOrder); ++i) {
        if (i < (KB ? KB : kb)) {
            D acc[4] = {};
            const D* const line = f + i * ldf;  // row i of F (solve) or column i (product)
            if constexpr (Inverse) {
                BATCHLAS_UNROLL_FULL
                for (int p0 = 0; p0 < i; p0 += V) {
                    const Pack<D> pk = *reinterpret_cast<const Pack<D>*>(line + p0);
                    BATCHLAS_UNROLL_FULL
                    for (int u = 0; u < V; ++u)
                        if (p0 + u < i) acc[(p0 + u) % 4] = sub_mul(acc[(p0 + u) % 4], pk.e[u], c.v[p0 + u]);
                }
            } else {
                BATCHLAS_UNROLL_FULL
                for (int p0 = (i + 1) / V * V; p0 < (KB ? KB : kDiagMaxOrder); p0 += V) {
                    if (p0 >= (KB ? KB : kb)) continue;
                    const Pack<D> pk = *reinterpret_cast<const Pack<D>*>(line + p0);
                    BATCHLAS_UNROLL_FULL
                    for (int u = 0; u < V; ++u)
                        if (p0 + u > i && p0 + u < (KB ? KB : kb))
                            acc[(p0 + u) % 4] = add_conj_mul(acc[(p0 + u) % 4], pk.e[u], c.v[p0 + u]);
                }
            }
            const D sum = add(add(acc[0], acc[1]), add(acc[2], acc[3]));
            const auto fii = sycl_device::dev_real(line[i]);
            c.v[i] = Inverse ? sycl_device::dev_div_real(add(c.v[i], sum), fii)
                             : add(sycl_device::dev_mul_real(c.v[i], fii), sum);
        }
    }
    return c;
}

template <bool Inverse, int KB, typename T, typename D>
void launch_diag(Queue& ctx, D* ap, int lda, std::ptrdiff_t stride_a, const D* bp, int ldb, std::ptrdiff_t stride_b,
                 int kb_rt, int batch, bool lower, D* hp, int ldh, std::ptrdiff_t stride_h, const std::uint8_t* mask) {
    const int kb = KB ? KB : kb_rt;
    const int ipw = diag_items_per_group<T>(kb), ldf = diag_ldf<T>(kb), lds = kb + 1;
    const int fper = kb * ldf, sper = kb * lds, groups = (batch + ipw - 1) / ipw;
    const int wg = (ipw * kb + 31) / 32 * 32;
    const std::size_t packs = (static_cast<std::size_t>(ipw) * (fper + sper) + kVec<D> - 1) / kVec<D>;
    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<Pack<D>, 1> slm(sycl::range<1>(packs), h);
        h.parallel_for<SygstDiagKernel<D, Inverse, KB>>(
            sycl::nd_range<1>(static_cast<std::size_t>(groups) * wg, wg), [=](sycl::nd_item<1> it) {
                const int t = static_cast<int>(it.get_local_id(0));
                const std::ptrdiff_t first = static_cast<std::ptrdiff_t>(it.get_group(0)) * ipw;
                D* const fbase = reinterpret_cast<D*>(&slm[0]);
                D* const sbase = fbase + ipw * fper;
                auto named = [=](int i, int c) { return lower ? i >= c : i <= c; };
                // Column-wise and coalesced: thread t owns row t % kb of columns t / kb, + wg / kb, ...
                // I2: a masked item is a dead partition, never loaded, computed or stored.
                const int row = t % kb, cstep = wg / kb;
                for (int idx = t / kb; idx < ipw * kb; idx += cstep) {
                    const int gi = idx / kb, c = idx - gi * kb;
                    const std::ptrdiff_t itm = first + gi;
                    if (row >= kb || itm >= batch || (mask && !mask[itm]) || !named(row, c)) continue;
                    sbase[gi * sper + row + c * lds] = ap[itm * stride_a + static_cast<std::ptrdiff_t>(c) * lda + row];
                    const D bv = bp[itm * stride_b + static_cast<std::ptrdiff_t>(c) * ldb + row];
                    D* const fg = fbase + gi * fper;  // F(row, c) = B(row, c) or F(c, row) = conj(B(row, c))
                    if (lower) fg[fidx<Inverse>(row, c, ldf)] = bv;
                    else fg[fidx<Inverse>(c, row, ldf)] = sycl_device::dev_conj(bv);
                }
                const int g = t / kb, j = t - g * kb;
                const bool active = g < ipw && first + g < batch && (!mask || mask[first + g]);
                const D* const f = fbase + (active ? g * fper : 0);
                D* const s = sbase + (active ? g * sper : 0);
                it.barrier(sycl::access::fence_space::local_space);
                Col<D> col{};
                BATCHLAS_UNROLL_FULL
                for (int i = 0; i < (KB ? KB : kDiagMaxOrder); ++i)
                    if (active && i < kb)
                        col.v[i] = i == j ? real_only(s[i + j * lds])
                                   : named(i, j) ? s[i + j * lds] : sycl_device::dev_conj(s[j + i * lds]);
                it.barrier(sycl::access::fence_space::local_space);
                if (active) col = apply<Inverse, KB>(col, f, ldf, kb);
                BATCHLAS_UNROLL_FULL
                for (int i = 0; i < (KB ? KB : kDiagMaxOrder); ++i)
                    if (active && i < kb) s[i + j * lds] = col.v[i];
                it.barrier(sycl::access::fence_space::local_space);
                // C = F^-1 X^H (or F^H X^H): X's conjugate transpose, row j of X as column j.
                BATCHLAS_UNROLL_FULL
                for (int i = 0; i < (KB ? KB : kDiagMaxOrder); ++i)
                    if (active && i < kb) col.v[i] = sycl_device::dev_conj(s[j + i * lds]);
                it.barrier(sycl::access::fence_space::local_space);
                if (active) col = apply<Inverse, KB>(col, f, ldf, kb);
                BATCHLAS_UNROLL_FULL
                for (int i = 0; i < (KB ? KB : kDiagMaxOrder); ++i)
                    if (active && i < kb) s[i + j * lds] = col.v[i];
                it.barrier(sycl::access::fence_space::local_space);
                for (int idx = t / kb; idx < ipw * kb; idx += cstep) {
                    const int gi = idx / kb, c = idx - gi * kb;
                    const std::ptrdiff_t itm = first + gi;
                    if (row >= kb || itm >= batch || (mask && !mask[itm])) continue;
                    const D* const sg = sbase + gi * sper;
                    const D v = row == c ? real_only(sg[row + c * lds])
                                : named(row, c) ? sg[row + c * lds] : sycl_device::dev_conj(sg[c + row * lds]);
                    if (hp) hp[itm * stride_h + static_cast<std::ptrdiff_t>(c) * ldh + row] = v;
                    if (named(row, c)) ap[itm * stride_a + static_cast<std::ptrdiff_t>(c) * lda + row] = v;
                }
            });
    });
}

}  // namespace

template <typename T>
Event sygst_diag(Queue& ctx, T* a, int lda, std::ptrdiff_t stride_a, const T* b, int ldb, std::ptrdiff_t stride_b,
                 int kb, int batch, int itype, Uplo uplo, T* full, int ldf, std::ptrdiff_t stride_f,
                 const std::uint8_t* mask) {
    using D = typename sycl_device::DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");
    if (kb < 1 || kb > kDiagMaxOrder || batch < 1) return ctx.get_event();
    D* const ap = reinterpret_cast<D*>(a);
    D* const hp = reinterpret_cast<D*>(full);
    const D* const bp = reinterpret_cast<const D*>(b);
    const bool lo = uplo == Uplo::Lower;
    // Compile-time orders for the blocks nb 16, 32, 64 produce and n = 8; any other order runs generic.
    auto run = [&]<bool Inv>() {
        auto go = [&]<int K>() {
            launch_diag<Inv, K, T>(ctx, ap, lda, stride_a, bp, ldb, stride_b, kb, batch, lo, hp, ldf, stride_f, mask);
        };
        if (kb == 32) go.template operator()<32>();
        else if (kb == 16) go.template operator()<16>();
        else if (kb == 8) go.template operator()<8>();
        else go.template operator()<0>();
    };
    if (itype == 1) run.template operator()<true>();
    else run.template operator()<false>();
    return ctx.get_event();
}

template <typename T>
Event sygst_mirror(Queue& ctx, const T* a, int lda, std::ptrdiff_t stride_a, int kb, int batch, Uplo uplo, T* full,
                   int ldf, std::ptrdiff_t stride_f, const std::uint8_t* mask) {
    using D = typename sycl_device::DevMap<T>::type;
    const D* const ap = reinterpret_cast<const D*>(a);
    D* const hp = reinterpret_cast<D*>(full);
    const bool lower = uplo == Uplo::Lower;
    const std::size_t per = static_cast<std::size_t>(kb) * static_cast<std::size_t>(kb);
    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<SygstMirrorKernel<D>>(sycl::range<1>(per * static_cast<std::size_t>(batch)), [=](sycl::item<1> it) {
            const std::size_t e = it.get_id(0);
            const auto itm = static_cast<std::ptrdiff_t>(e / per);
            if (mask && !mask[itm]) return;  // I3
            const int r = static_cast<int>(e % per), i = r % kb, c = r / kb;
            const bool own = lower ? i >= c : i <= c;
            const D* const src = ap + itm * stride_a;
            const D v = own ? src[static_cast<std::ptrdiff_t>(c) * lda + i]
                            : sycl_device::dev_conj(src[static_cast<std::ptrdiff_t>(i) * lda + c]);
            hp[itm * stride_f + static_cast<std::ptrdiff_t>(c) * ldf + i] = i == c ? real_only(v) : v;
        });
    });
    return ctx.get_event();
}

#define BATCHLAS_SYGST_DIAG_INSTANTIATE(T)                                                                  \
    template Event sygst_diag<T>(Queue&, T*, int, std::ptrdiff_t, const T*, int, std::ptrdiff_t, int, int, int, \
                                 Uplo, T*, int, std::ptrdiff_t, const std::uint8_t*);                          \
    template Event sygst_mirror<T>(Queue&, const T*, int, std::ptrdiff_t, int, int, Uplo, T*, int, std::ptrdiff_t, \
                                   const std::uint8_t*);

BATCHLAS_SYGST_DIAG_INSTANTIATE(float)
BATCHLAS_SYGST_DIAG_INSTANTIATE(double)
BATCHLAS_SYGST_DIAG_INSTANTIATE(std::complex<float>)
BATCHLAS_SYGST_DIAG_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_SYGST_DIAG_INSTANTIATE

}  // namespace batchlas::sycl_sygst
