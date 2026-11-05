// sygst blocked's diagonal block (sygst_native.hh). One work-item per column of a kb x kb block, the
// column in registers; the factor F and a transpose buffer per item in shared memory, F stored so each
// pass reads its rows (or columns) as 16-byte packs. Each pass is a triangular solve with F (itype 1)
// or a product with F^H (2, 3), applied to A and then to the conjugate transpose of the result, which
// is the Hermitian C itself. Orders 32, 16 and 8 have compile-time instances.
// evidence: docs/perf/sygvd.md#sygst-the-blocked-family

#include <src/ops/sygst/sygst_native.hh>

#include <src/ops/sygst/sygst_v2.hh>
#include <src/sycl/device_scalar.hh>
#include <src/queue.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace batchlas::sycl_sygst {

namespace {

namespace v2 = batchlas::device;

template <typename D, bool Inverse, int KB> class SygstDiagKernel;
template <typename D> class SygstMirrorKernel;

template <typename D>
using Col = v2::Frag<D, kDiagMaxOrder>;  // one column of the block, in registers
template <typename D>
using Pack = dla::Pack16<D>;  // F is read as aligned 16-byte packs (diag_ldf pads its lines)
template <typename D>
constexpr int kVec = dla::pack16_elems<D>;
// Every element of the 16-byte-aligned SLM packs is aligned to min(16, sizeof(D)): said through the
// operand base, since a view's pointer member drops the alignment derived from &slm[0] (a complex
// S column would go out as two 8-byte shared stores per element instead of one 16-byte store).
template <typename D>
constexpr int kSlmAlign = sizeof(D) < 16 ? static_cast<int>(sizeof(D)) : 16;
template <typename D>
using Slm = v2::Aligned<D, kSlmAlign<D>>;

template <int KB>
using KbExt = std::conditional_t<(KB > 0), v2::Ext<KB>, v2::Dyn>;
template <int KB>
inline KbExt<KB> kb_ext(int kb) {
    if constexpr (KB > 0) return {};
    else return v2::Dyn{kb};
}

// F(r, q) as stored: F^T column-major for the solve (column i of the store is row i of F),
// F column-major for the product.
template <bool Inverse, class M>
inline auto& fat(const M& fs, int r, int q) { return Inverse ? fs(q, r) : fs(r, q); }

// F^-1 c (itype 1, ?TRSV with the stored F^T) or F^H c (2, 3, ?TRMV). A plain inline function as
// before the port, so the compiler's inlining choice for the two calls is unchanged.
template <bool Inverse, int KB, typename D>
inline Col<D> apply(Col<D> c, const D* f, int ldf, int kb) {
    const KbExt<KB> kn = kb_ext<KB>(kb);
    const v2::Mat<const D, KbExt<KB>, KbExt<KB>, v2::Dyn, v2::Mem::Local> fs{f, kn, kn, v2::Dyn{ldf}};
    if constexpr (Inverse) dla::trsv_packed<Uplo::Upper, Transpose::Trans>(v2::Thread{}, fs, c);
    else dla::trmv_packed<Uplo::Lower, Transpose::ConjTrans>(v2::Thread{}, fs, c);
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
                using SMat = v2::Mat<D, KbExt<KB>, KbExt<KB>, v2::Dyn, v2::Mem::Local, v2::Cyclic, Slm<D>>;
                const int live = KB ? KB : kb;
                const KbExt<KB> kn = kb_ext<KB>(kb);
                const Uplo ul = lower ? Uplo::Lower : Uplo::Upper;
                const int t = static_cast<int>(it.get_local_id(0));
                const std::ptrdiff_t first = static_cast<std::ptrdiff_t>(it.get_group(0)) * ipw;
                D* const fbase = reinterpret_cast<D*>(&slm[0]);
                D* const sbase = fbase + ipw * fper;
                auto a_of = [=](std::ptrdiff_t itm) { return v2::mat(ap + itm * stride_a, kb, kb, lda); };
                auto s_of = [=](int gi) { return v2::mat_aligned<kSlmAlign<D>, v2::Mem::Local>(sbase + gi * sper, kb, kb, lds); };
                // ?LACPY of A's uplo triangle into S and of B's into F (conjugate-transposed from U),
                // column-wise and coalesced. I2: a masked item is a dead partition, never loaded,
                // computed or stored.
                dla::for_each_tile_col(t, wg, kb, ipw, [&](int gi, int row, int c) BATCHLAS_DLA_BODY {
                    const std::ptrdiff_t itm = first + gi;
                    if (row >= kb || itm >= batch || (mask && !mask[itm]) || !dla::in_triangle(ul, row, c)) return;
                    s_of(gi)(row, c) = a_of(itm)(row, c);
                    const D bv = v2::mat(bp + itm * stride_b, kb, kb, ldb)(row, c);
                    const auto fs = v2::mat_aligned<kSlmAlign<D>, v2::Mem::Local>(fbase + gi * fper, kb, kb, ldf);
                    if (lower) fat<Inverse>(fs, row, c) = bv;
                    else fat<Inverse>(fs, c, row) = v2::Arith<D>::conj(bv);
                });
                const int g = t / kb, j = t - g * kb;
                const bool active = g < ipw && first + g < batch && (!mask || mask[first + g]);
                const D* const f = fbase + (active ? g * fper : 0);
                const SMat s{Slm<D>{sbase + (active ? g * sper : 0)}, kn, kn, v2::Dyn{lds}};
                it.barrier(sycl::access::fence_space::local_space);
                // X = op(F) A, column j: the Hermitian A's column, then F^-1 or F^H.
                Col<D> col{};
                if (active) v2::load(col, dla::herm(s, ul).col(j), live);
                it.barrier(sycl::access::fence_space::local_space);
                if (active) col = apply<Inverse, KB>(col, f, ldf, kb);
                if (active) v2::store(col, s.col(j), live);
                it.barrier(sycl::access::fence_space::local_space);
                // C = op(F) X^H: X's conjugate transpose, row j of X as column j.
                if (active) v2::load(col, v2::conj_view(s.row(j)), live);
                it.barrier(sycl::access::fence_space::local_space);
                if (active) col = apply<Inverse, KB>(col, f, ldf, kb);
                if (active) v2::store(col, s.col(j), live);
                it.barrier(sycl::access::fence_space::local_space);
                // The Hermitian C (real diagonal) to `full` and its uplo triangle back to A.
                dla::for_each_tile_col(t, wg, kb, ipw, [&](int gi, int row, int c) BATCHLAS_DLA_BODY {
                    const std::ptrdiff_t itm = first + gi;
                    if (row >= kb || itm >= batch || (mask && !mask[itm])) return;
                    const D v = dla::herm(s_of(gi), ul)(row, c);
                    if (hp) v2::mat(hp + itm * stride_h, kb, kb, ldh)(row, c) = v;
                    if (dla::in_triangle(ul, row, c)) a_of(itm)(row, c) = v;
                });
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
    const std::size_t per = static_cast<std::size_t>(kb) * static_cast<std::size_t>(kb);
    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<SygstMirrorKernel<D>>(sycl::range<1>(per * static_cast<std::size_t>(batch)), [=](sycl::item<1> it) {
            const std::size_t e = it.get_id(0);
            const auto itm = static_cast<std::ptrdiff_t>(e / per);
            if (mask && !mask[itm]) return;  // I3
            const int r = static_cast<int>(e % per), i = r % kb, c = r / kb;
            // H = the Hermitian A, real diagonal. H is named first: the capture order is the kernel's
            // parameter order (hp, then ap), as in the hand kernel.
            const auto H = v2::mat(hp + itm * stride_f, kb, kb, ldf);
            v2::lacpy(v2::ElemAt<true>{i, c}, dla::herm(v2::mat(ap + itm * stride_a, kb, kb, lda), uplo), H);
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
