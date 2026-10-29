// SPDX-License-Identifier: MIT
// syfn's element-wise kernels: info, zero, weights (the column scales) and the column scale.
// evidence: docs/perf/syfn.md#syfn-composition

#include <batchlas/backend_config.h>

#include <src/ops/syfn/syfn_native.hh>

#include <src/queue.hh>
#include <src/util/template-instantiations.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>

namespace batchlas {
namespace ops::syfn {

template <Backend B, class T>
struct WeightsKernel {};
template <Backend B, class T>
struct InfoKernel {};
template <Backend B, class T>
struct ScaleKernel {};
template <Backend B, class T>
struct ZeroKernel {};

template <class R>
inline bool out_of_domain(SyfnKind k, R w) {
    switch (k) {
        case SyfnKind::InvSqrt: return w <= R(0);
        case SyfnKind::Sqrt: return w < R(0);
        case SyfnKind::Inverse: return w == R(0);
        default: return false;
    }
}

// The column scale: sqrt(f(w)) for the rank-k form, f(w) otherwise; 0 out of the domain.
template <class R>
inline R column_scale(SyfnKind k, R w, R weight) {
    if (out_of_domain(k, w)) return R(0);
    switch (k) {
        case SyfnKind::InvSqrt: return R(1) / sycl::sqrt(sycl::sqrt(w));
        case SyfnKind::Sqrt: return sycl::sqrt(sycl::sqrt(w));
        case SyfnKind::Exp: return sycl::exp(w * R(0.5));
        case SyfnKind::Inverse: return R(1) / w;
        default: return weight;
    }
}

template <Backend B, class T>
void launch_info(Queue& q, const MV<T>& V, Span<float_t<T>> W, SyfnKind kind, Span<int32_t> info,
                 Span<int32_t> syev_info, const std::uint8_t* mask) {
    using R = float_t<T>;
    const int64_t n = V.rows(), m = V.cols(), batch = V.batch_size();
    const R* wp = W.data();
    int32_t* ip = info.data();
    const int32_t* sp = syev_info.empty() ? nullptr : syev_info.data();
    q->parallel_for<InfoKernel<B, T>>(sycl::range<1>(size_t(batch)), [=](sycl::id<1> id) {
        const int64_t b = int64_t(id[0]);
        if (mask && !mask[b]) return;  // I3
        if (sp && sp[b] != 0) {
            ip[b] = int32_t(n) + sp[b];
            return;
        }
        int32_t bad = 0;
        for (int64_t j = 0; j < m; ++j) bad += out_of_domain(kind, wp[b * m + j]) ? 1 : 0;
        ip[b] = bad;
    });
}

template <Backend B, class T>
Event launch_zero(Queue& q, const MV<T>& V, const MV<T>& F, const std::uint8_t* mask) {
    const int64_t n = V.rows(), batch = V.batch_size();
    T* fp = F.data_ptr();
    const int64_t ldf = F.ld(), sf = F.stride();
    q->parallel_for<ZeroKernel<B, T>>(sycl::range<1>(size_t(n * n * batch)), [=](sycl::id<1> id) {
        const int64_t l = int64_t(id[0]), b = l / (n * n), e = l % (n * n);
        if (mask && !mask[b]) return;  // I3
        fp[b * sf + (e / n) * ldf + e % n] = T(0);
    });
    return q.get_event();
}

template <Backend B, class T>
void launch_scale(Queue& q, const MV<T>& V, Span<float_t<T>> W, SyfnFunction<float_t<T>> fn, const Core<T>& ws,
                  const std::uint8_t* mask) {
    using R = float_t<T>;
    const int64_t n = V.rows(), m = V.cols(), batch = V.batch_size();
    const SyfnKind kind = fn.kind;
    const R* wp = W.data();
    const R* up = fn.weights.data();
    R* gp = ws.g.data();
    q->parallel_for<WeightsKernel<B, T>>(sycl::range<1>(size_t(m * batch)), [=](sycl::id<1> id) {
        const size_t k = id[0];
        if (mask && !mask[k / size_t(m)]) return;  // I3
        gp[k] = column_scale(kind, wp[k], kind == SyfnKind::Weights ? up[k] : R(0));
    });
    const T* vp = V.data_ptr();
    const int64_t ldv = V.ld(), sv = V.stride();
    T* yp = ws.y.data();
    q->parallel_for<ScaleKernel<B, T>>(sycl::range<1>(size_t(n * m * batch)), [=](sycl::id<1> id) {
        const int64_t l = int64_t(id[0]), b = l / (n * m), e = l % (n * m), i = e % n, j = e / n;
        if (mask && !mask[b]) return;  // I3
        yp[l] = vp[b * sv + j * ldv + i] * gp[b * m + j];
    });
}

#define SYFN_KERNELS_INSTANTIATE(B_, fp)                                                                       \
    template void launch_info<B_, BATCHLAS_UNPAREN fp>(Queue&, const MV<BATCHLAS_UNPAREN fp>&,                 \
                                                       Span<float_t<BATCHLAS_UNPAREN fp>>, SyfnKind,           \
                                                       Span<int32_t>, Span<int32_t>, const std::uint8_t*);     \
    template Event launch_zero<B_, BATCHLAS_UNPAREN fp>(Queue&, const MV<BATCHLAS_UNPAREN fp>&,                \
                                                        const MV<BATCHLAS_UNPAREN fp>&, const std::uint8_t*);  \
    template void launch_scale<B_, BATCHLAS_UNPAREN fp>(Queue&, const MV<BATCHLAS_UNPAREN fp>&,                \
                                                        Span<float_t<BATCHLAS_UNPAREN fp>>,                    \
                                                        SyfnFunction<float_t<BATCHLAS_UNPAREN fp>>,            \
                                                        const Core<BATCHLAS_UNPAREN fp>&, const std::uint8_t*);
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYFN_KERNELS_INSTANTIATE)
#undef SYFN_KERNELS_INSTANTIATE

}  // namespace ops::syfn
}  // namespace batchlas
