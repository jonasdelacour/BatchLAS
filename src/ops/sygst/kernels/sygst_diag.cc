// sygst blocked's diagonal block (sygst_native.hh): a thin launcher over the public ?sygs2 body,
// device::sygs2_group (include/batchlas/device/lapack/sygst.hh), ipw blocks per work-group. Orders
// 32, 16 and 8 have compile-time instances.
// evidence: docs/perf/sygvd.md#sygst-the-blocked-family

#include <src/ops/sygst/sygst_native.hh>

#include <batchlas/device/lapack/sygst.hh>
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
using Pack = v2::sygst_detail::Pack16<D>;  // F is read as aligned 16-byte packs (diag_ldf pads its lines)
template <typename D>
constexpr int kVec = v2::sygst_detail::pack16_elems<D>;

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
                // The arguments name the captures in the hand kernel's order (its parameter order).
                // I2: a masked item is a dead block, never loaded, computed or stored.
                v2::sygs2_group<Inverse, KB>(
                    v2::RtGroup<>{it}, static_cast<int>(it.get_local_id(0)), kb, lower, it.get_group(0), ipw,
                    reinterpret_cast<D*>(&slm[0]), fper,
                    [=](std::ptrdiff_t itm) { return v2::mat(ap + itm * stride_a, kb, kb, lda); }, sper, lds, wg,
                    [=](std::ptrdiff_t itm) BATCHLAS_DLA_BODY { return itm >= batch || (mask && !mask[itm]); },
                    [=](std::ptrdiff_t itm) BATCHLAS_DLA_BODY { return v2::mat(bp + itm * stride_b, kb, kb, ldb); }, ldf,
                    [=](std::ptrdiff_t itm, int row, int c, const D& v) BATCHLAS_DLA_BODY {
                        if (hp) v2::mat(hp + itm * stride_h, kb, kb, ldh)(row, c) = v;
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
            v2::lacpy(v2::ElemAt<true>{i, c}, v2::sygst_detail::herm(v2::mat(ap + itm * stride_a, kb, kb, lda), uplo), H);
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
