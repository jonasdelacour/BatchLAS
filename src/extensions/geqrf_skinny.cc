// Native batched GEQRF: the CTA tier's SKINNY leg, n <= 8 panels held in one sub-group's
// registers (geqrf_skinny_device.hh). It is a leg of the CTA tier, not a tier: the route is
// still native:cta, and geqrf_cta_dispatch decides between this and the resident leaf.
// evidence: docs/perf/blackwell.md#geqrf-the-skinny-register-leg

#include "geqrf_native.hh"
#include "geqrf_skinny_device.hh"

#include "../queue.hh"

#include <batchlas/blas/dispatch/route_geqrf.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

// Outside the anonymous namespace: a kernel name may not name an internal-linkage entity.
namespace batchlas {
template <typename D, int N, int RP, int P> struct GeqrfSkinnyKernel;
}

namespace batchlas {

namespace gn = ::batchlas::geqrf_native;

// A FUNCTOR so the launch bound can be spelled (it is refused on a lambda). MinBlocks = 1:
// any tighter cap spills the RP >= 8 cells to the stack (ptxas sm_120).
// evidence: docs/perf/blackwell.md#geqrf-the-skinny-register-leg
template <typename D, int N, int RP, int P>
struct GeqrfSkinnyKernel {
    D* ap;
    D* tp;
    int m;
    int n;
    int batch;
    std::ptrdiff_t ld;
    std::ptrdiff_t stride;
    int tau_bs;
    int tau_off;

    [[sycl::reqd_sub_group_size(32), BATCHLAS_LAUNCH_BOUNDS(gn::kGeqrfSkinnyWg, 1)]]
    void operator()(sycl::nd_item<1> it) const {
        constexpr int kMps = 32 / P;   // matrices per sub-group
        constexpr int kSgs = gn::kGeqrfSkinnyWg / 32;
        const auto sg = it.get_sub_group();
        const int sgid = static_cast<int>(it.get_group_linear_id()) * kSgs +
                         static_cast<int>(sg.get_group_linear_id());
        // Sub-group-uniform, so legal; a partially live sub-group CLAMPS instead, because
        // its dead partitions still sit in every shuffle's mask.
        if (sgid * kMps >= batch) return;
        const int prob = sgid * kMps + static_cast<int>(sg.get_local_linear_id()) / P;
        const bool live = prob < batch;
        const int b = live ? prob : 0;
        gn::geqr2_skinny_device<D, N, RP, P>(
            sg, ap + static_cast<std::ptrdiff_t>(b) * stride, ld, m, n,
            tp + static_cast<std::ptrdiff_t>(b) * tau_bs + tau_off, live);
    }
};

namespace sycl_geqrf {

namespace {

template <typename T>
constexpr bool geqrf_skinny_type() {
    return std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>;
}

// Row capacity: cfloat stops at 256 rows, where its 16-slot bucket measured slower than the
// resident leaf (257x4, 300x3) and would hold 256 live registers a lane at N = 8.
template <typename T>
constexpr int geqrf_skinny_rows(int N) {
    static_cast<void>(N);
    return std::is_same_v<T, std::complex<float>> ? 256 : 512;
}

template <typename T, int N, int RP, int P>
Event geqrf_skinny_launch_n(Queue& ctx, T* a_ptr, int ld, int stride, int m, int n, int batch,
                            T* tau_ptr, int tau_bs, int tau_off) {
    using D = typename sycl_device::DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");
    constexpr int kMpw = (gn::kGeqrfSkinnyWg / 32) * (32 / P);
    const int num_wg = (batch + kMpw - 1) / kMpw;
    const GeqrfSkinnyKernel<D, N, RP, P> k{
        reinterpret_cast<D*>(a_ptr), reinterpret_cast<D*>(tau_ptr), m, n, batch,
        static_cast<std::ptrdiff_t>(ld), static_cast<std::ptrdiff_t>(stride), tau_bs, tau_off};
    ctx->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_wg) *
                                                        gn::kGeqrfSkinnyWg),
                                         sycl::range<1>(gn::kGeqrfSkinnyWg)),
                       k);
    });
    return ctx.get_event();
}

template <typename T, int N>
Event geqrf_skinny_launch_rows(Queue& ctx, T* a_ptr, int ld, int stride, int m, int n,
                               int batch, T* tau_ptr, int tau_bs, int tau_off) {
#define BATCHLAS_SKINNY_ARGS ctx, a_ptr, ld, stride, m, n, batch, tau_ptr, tau_bs, tau_off
    if (m <= 8) return geqrf_skinny_launch_n<T, N, 1, 8>(BATCHLAS_SKINNY_ARGS);
    if (m <= 16) return geqrf_skinny_launch_n<T, N, 1, 16>(BATCHLAS_SKINNY_ARGS);
    if (m <= 32) return geqrf_skinny_launch_n<T, N, 1, 32>(BATCHLAS_SKINNY_ARGS);
    if (m <= 64) return geqrf_skinny_launch_n<T, N, 2, 32>(BATCHLAS_SKINNY_ARGS);
    if (m <= 128) return geqrf_skinny_launch_n<T, N, 4, 32>(BATCHLAS_SKINNY_ARGS);
    if (m <= 256) return geqrf_skinny_launch_n<T, N, 8, 32>(BATCHLAS_SKINNY_ARGS);
    if constexpr (geqrf_skinny_rows<T>(N) >= 512) {
        if (m <= 512) return geqrf_skinny_launch_n<T, N, 16, 32>(BATCHLAS_SKINNY_ARGS);
    }
#undef BATCHLAS_SKINNY_ARGS
    throw batchlas::internal_error("geqrf_skinny: no row bucket for m = " + std::to_string(m));
}

}  // namespace

template <typename T>
bool geqrf_skinny_fits(int m, int n) {
    if constexpr (!geqrf_skinny_type<T>()) {
        static_cast<void>(m);
        static_cast<void>(n);
        return false;
    } else {
        if (n < 1 || n > gn::kGeqrfSkinnyMaxCols || m < n) return false;
        return m <= geqrf_skinny_rows<T>(n <= 4 ? 4 : 8);
    }
}

template <typename T>
bool geqrf_skinny_preferred(int cuda_cc, int m, int n) {
    return geqrf_skinny_fits<T>(m, n) && dispatch::geqrf_skinny_window<T>(cuda_cc, m, n);
}

template <typename T>
Event geqrf_skinny_launch(Queue& ctx, T* a_ptr, int ld, int stride, int m, int n, int batch,
                          T* tau_ptr, int tau_bs, int tau_off) {
    if (!geqrf_skinny_fits<T>(m, n)) {
        throw batchlas::invalid_argument(
            "geqrf_skinny: a " + std::to_string(m) + " x " + std::to_string(n) +
            " panel does not fit the skinny leg (float/cfloat, n <= 8, n <= m, m <= 512; "
            "cfloat m <= 256)");
    }
    if constexpr (geqrf_skinny_type<T>()) {
        if (n <= 4) {
            return geqrf_skinny_launch_rows<T, 4>(ctx, a_ptr, ld, stride, m, n, batch, tau_ptr,
                                                  tau_bs, tau_off);
        }
        return geqrf_skinny_launch_rows<T, 8>(ctx, a_ptr, ld, stride, m, n, batch, tau_ptr,
                                              tau_bs, tau_off);
    } else {
        static_cast<void>(ctx);
        static_cast<void>(a_ptr);
        static_cast<void>(tau_ptr);
        throw batchlas::unsupported("geqrf_skinny: no skinny leg for this scalar type");
    }
}

#define BATCHLAS_GEQRF_SKINNY_INSTANTIATE(T)                                                  \
    template bool geqrf_skinny_fits<T>(int, int);                                             \
    template bool geqrf_skinny_preferred<T>(int, int, int);                                   \
    template Event geqrf_skinny_launch<T>(Queue&, T*, int, int, int, int, int, T*, int, int);

BATCHLAS_GEQRF_SKINNY_INSTANTIATE(float)
BATCHLAS_GEQRF_SKINNY_INSTANTIATE(double)
BATCHLAS_GEQRF_SKINNY_INSTANTIATE(std::complex<float>)
BATCHLAS_GEQRF_SKINNY_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GEQRF_SKINNY_INSTANTIATE

}  // namespace sycl_geqrf
}  // namespace batchlas
