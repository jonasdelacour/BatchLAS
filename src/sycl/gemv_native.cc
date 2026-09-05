// Native batched GEMV: five kernel bodies, the work-group ladder they share,
// and the capability flags as explicit specialisations in this same TU. Read
// src/sycl/gemv_native.hh first for the layout premise and the contracts.
// evidence: docs/perf/gemv.md#the-five-kernel-bodies

#include "gemv_native.hh"

#include "../queue.hh"
#include "device_scalar.hh"

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace batchlas::sycl_gemv {

using sycl_device::DevMap;
using sycl_device::dev_conj;
using sycl_device::dev_is_complex_v;
using sycl_device::fma_acc;

namespace {

// The work-group ladder. The grid is FLATTENED to 1-D over (out_len * batch),
// b = gid / out_len, so the work-group count never depends on the batch alone.
inline int gemv_wg_ladder(int64_t work_units, int max_wg, int cu,
                          int units_per_wg_shift) {
    // units_per_wg_shift: 0 = one work-ITEM per unit, 5 = one 32-lane SUB-GROUP.
    int wg = 32;
    for (int cand : {256, 128, 64, 32}) {
        if (cand > max_wg) continue;
        wg = cand;
        const int64_t per_wg = cand >> units_per_wg_shift;
        if (per_wg < 1) continue;
        const int64_t groups = (work_units + per_wg - 1) / per_wg;
        if (groups >= static_cast<int64_t>(4) * cu) break;
    }
    // `wg` is still 32 if no candidate was admissible, which may exceed max_wg.
    if (max_wg > 0 && wg > max_wg) wg = max_wg;
    return wg;
}

// A sub-group sum, hand-rolled rather than sycl::reduce_over_group, which puts
// static shared memory into the kernel. Afterwards LANE 0 alone holds the total.
template <typename SG, typename D>
inline D sg_sum(const SG& sg, D v) {
    if constexpr (dev_is_complex_v<D>) {
        auto re = v.re, im = v.im;
        for (int off = 16; off > 0; off >>= 1) {
            re += sycl::shift_group_left(sg, re, off);
            im += sycl::shift_group_left(sg, im, off);
        }
        return D{re, im};
    } else {
        for (int off = 16; off > 0; off >>= 1) v += sycl::shift_group_left(sg, v, off);
        return v;
    }
}

template <typename T> class GemvDirectNKernel;
template <typename T> class GemvDirectTKernel;
template <typename T> class GemvCtaTKernel;
template <typename T, int W> class GemvSegNKernel;
template <typename T, int W> class GemvSegTKernel;

// Body 4's segment width: the largest power of two W with W * out_len <= 32, so
// one sub-group covers a whole output vector; W == 1 gates body 4 off. W is a
// TEMPLATE parameter -- with a runtime trip count and stride the fold stops
// unrolling and this body is SLOWER than body 1.
// evidence: docs/perf/gemv.md#the-body-4-gate
inline int gemv_seg_width(int out_len) {
    if (out_len <= 0) return 1;
    int w = 1;
    while (w * 2 * out_len <= 32) w *= 2;
    return w;
}

// Body 5's maximum segment width -- W outputs per sub-group, L = 32/W lanes each.
// The table selects 8 and 4; W = 2 exists only so BATCHLAS_GEMV_SEGT=2 means W = 2.
inline constexpr int kGemvSegTransMaxW = 8;

// GATE 1 -- the admitted band per type, ON red_len() and NEVER out_len(): under
// Trans red_len() == A.rows(), and a predicate on the other extent inverts it.
// evidence: docs/perf/gemv.md#the-sub-route-gates
template <typename T> inline constexpr int kGemvSegTransMaxRedLen = 0;
template <> inline constexpr int kGemvSegTransMaxRedLen<float> = 32;
template <> inline constexpr int kGemvSegTransMaxRedLen<std::complex<float>> = 16;
template <> inline constexpr int kGemvSegTransMaxRedLen<double> = 48;
template <> inline constexpr int kGemvSegTransMaxRedLen<std::complex<double>> = 64;

// GATE 2 -- which W: 8 at the short end of the admitted band, 4 at the long end.
// evidence: docs/perf/gemv.md#the-body-5-gates
template <typename T> inline constexpr int kGemvSegTransW8MaxRedLen = 0;
template <> inline constexpr int kGemvSegTransW8MaxRedLen<float> = 24;
template <> inline constexpr int kGemvSegTransW8MaxRedLen<std::complex<float>> = 16;
template <> inline constexpr int kGemvSegTransW8MaxRedLen<double> = 32;
template <> inline constexpr int kGemvSegTransW8MaxRedLen<std::complex<double>> = 32;

// GATE 3 -- a floor on the LAUNCH, per W, read off the device: body 5 launches W
// times fewer sub-groups than body 3, so a small launch cannot spare the
// parallelism.  evidence: docs/perf/gemv.md#the-body-5-gates
inline int gemv_seg_trans_min_items(int cu, int w) {
    const int c = (cu > 0) ? cu : 1;
    return (w >= 8 ? 16 : 64) * c;
}

// The spelling knob -- bodies 3 and 5 are one route, so BATCHLAS_GEMV_ROUTE
// cannot separate them. NOTHING IS LATCHED: a getenv cached in a function-local
// static makes a later setenv invisible, and the test then passes green on the
// default arm.
//   BATCHLAS_GEMV_SEGT = off | auto (default) | 2|4|8 (force body 5 at that W)
enum class SegTMode { kAuto, kOff, kForce2, kForce4, kForce8 };

inline SegTMode gemv_segt_mode() {
    const char* const s = std::getenv("BATCHLAS_GEMV_SEGT");
    if (s == nullptr) return SegTMode::kAuto;
    if (std::strcmp(s, "off") == 0) return SegTMode::kOff;
    if (std::strcmp(s, "2") == 0) return SegTMode::kForce2;
    if (std::strcmp(s, "4") == 0) return SegTMode::kForce4;
    if (std::strcmp(s, "8") == 0) return SegTMode::kForce8;
    return SegTMode::kAuto;
}

template <typename T>
inline constexpr bool kGemvSegTransEmit =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;

// The width decision in ONE place: 1 means body 3, W >= 2 means body 5 at that W.
template <typename T>
inline int gemv_seg_trans_width(SegTMode mode, int red_len, int64_t items, int cu) {
    if constexpr (!kGemvSegTransEmit<T>) {
        static_cast<void>(mode);
        static_cast<void>(red_len);
        static_cast<void>(items);
        static_cast<void>(cu);
        return 1;
    } else {
        // The forced spellings bypass all three gates on purpose.
        switch (mode) {
            case SegTMode::kOff:    return 1;
            case SegTMode::kForce2: return 2;
            case SegTMode::kForce4: return 4;
            case SegTMode::kForce8: return 8;
            case SegTMode::kAuto:   break;
        }
        static_assert(kGemvSegTransW8MaxRedLen<T> <= kGemvSegTransMaxRedLen<T>,
                      "the W = 8 band must lie inside the admitted band");
        static_assert(kGemvSegTransMaxW >= 8, "the W table selects 8 and 4");
        if (red_len <= 0) return 1;
        if (red_len > kGemvSegTransMaxRedLen<T>) return 1;
        const int w = (red_len <= kGemvSegTransW8MaxRedLen<T>) ? 8 : 4;
        if (items < gemv_seg_trans_min_items(cu, w)) return 1;
        return w;
    }
}

// Reference ?GEMV semantics: y is left COMPLETELY UNTOUCHED -- in particular NOT
// scaled by beta when the reduction length is 0.
template <typename T>
inline bool gemv_quick_return(int m, int n, T alpha, T beta) {
    return m == 0 || n == 0 || (alpha == T(0) && beta == T(1));
}

// BODY 1 -- {Native, Direct}, transA == NoTrans, one work-item per output row:
//   y_i = alpha * sum_j A[i + j*ld] * x[j*xinc] + beta * y[i*yinc]
// Coalesced ONLY FOR out_len >= 32: below the warp width a warp straddles
// 32/out_len batch items, and out_len*batch is the only parallel extent. Body 4
// fixes that; this one then serves devices with no 32-lane sub-group.
// evidence: docs/perf/gemv.md#the-body-4-gate
template <typename T>
Event gemv_direct_notrans(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& A,
                          const VectorView<T>& X,
                          const VectorView<T>& Y,
                          T alpha, T beta) {
    using D = typename DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    const int m = A.rows();
    const int n = A.cols();
    const int batch = A.batch_size();

    const int64_t items = static_cast<int64_t>(m) * batch;

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    const int wg = gemv_wg_ladder(items, max_wg, cu, /*units_per_wg_shift=*/0);
    const int64_t groups = (items + wg - 1) / wg;

    ctx->submit([&](sycl::handler& h) {
        // NO __restrict__ ON ANY OF THESE: ortho.cc passes views into the same
        // allocation, and __restrict__ is a promise about the object.
        const D* a_ptr = reinterpret_cast<const D*>(A.data_ptr());
        const D* x_ptr = reinterpret_cast<const D*>(X.data_ptr());
        D* y_ptr = reinterpret_cast<D*>(Y.data_ptr());

        const int64_t lda = A.ld();
        const int64_t stride_a = A.stride();
        const int64_t xinc = X.inc();
        const int64_t stride_x = X.stride();
        const int64_t yinc = Y.inc();
        const int64_t stride_y = Y.stride();

        D alpha_d, beta_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));
        __builtin_memcpy(&beta_d, &beta, sizeof(D));

        // `beta_zero` is what keeps y unread when beta == 0; y may be uninitialised.
        const bool alpha_zero = (alpha == T(0));
        const bool beta_zero = (beta == T(0));

        const int out_len = m;
        const int red_len = n;
        const int64_t total = items;

        h.parallel_for<GemvDirectNKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
                if (gid >= total) return;

                const int64_t b = gid / out_len;
                const int i = static_cast<int>(gid - b * out_len);

                const D* Ab = a_ptr + b * stride_a;
                const D* xb = x_ptr + b * stride_x;
                D* yb = y_ptr + b * stride_y;

                D acc{};
                if (!alpha_zero) {
                    for (int j = 0; j < red_len; ++j) {
                        fma_acc(acc, Ab[i + static_cast<int64_t>(j) * lda],
                                xb[static_cast<int64_t>(j) * xinc]);
                    }
                }

                // fma into an exact zero addend is the product, rounded once.
                D out{};
                fma_acc(out, alpha_d, acc);
                if (!beta_zero) fma_acc(out, beta_d, yb[static_cast<int64_t>(i) * yinc]);
                yb[static_cast<int64_t>(i) * yinc] = out;
            });
    });

    return ctx.get_event();
}

// BODY 4 -- {Native, Direct}, transA == NoTrans, SHORT OUTPUT. Body 1's route and
// answer by a different decomposition, so the choice is a property of the device
// and the shape, deliberately invisible to the routing vocabulary. W lanes per
// output, one sub-group per batch item: lane l takes i = l % m, jsub = l / m, so
// lanes l and l+m fold at stride m.
//
// THE FOLD IS CLOSED, which is why the inactive lanes are harmless: lane i draws
// only from lanes i + m*t, t < W, all below m*W <= 32. Lanes at or above m*W may
// shift past lane 31 and read an unspecified value, but nothing they produce is
// read -- `jsub < W` below is a work saving, not a correctness condition.
template <typename T, int W>
Event gemv_seg_notrans(Queue& ctx,
                       const MatrixView<T, MatrixFormat::Dense>& A,
                       const VectorView<T>& X,
                       const VectorView<T>& Y,
                       T alpha, T beta) {
    using D = typename DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    constexpr int kSg = 32;

    const int m = A.rows();
    const int n = A.cols();
    const int batch = A.batch_size();

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    const int wg = gemv_wg_ladder(batch, max_wg, cu, /*units_per_wg_shift=*/5);
    const int sgs_per_wg = (wg / kSg) > 0 ? (wg / kSg) : 1;
    const int64_t groups = (static_cast<int64_t>(batch) + sgs_per_wg - 1) / sgs_per_wg;

    ctx->submit([&](sycl::handler& h) {
        // NO __restrict__, for body 1's aliasing reason.
        const D* a_ptr = reinterpret_cast<const D*>(A.data_ptr());
        const D* x_ptr = reinterpret_cast<const D*>(X.data_ptr());
        D* y_ptr = reinterpret_cast<D*>(Y.data_ptr());

        const int64_t lda = A.ld();
        const int64_t stride_a = A.stride();
        const int64_t xinc = X.inc();
        const int64_t stride_x = X.stride();
        const int64_t yinc = Y.inc();
        const int64_t stride_y = Y.stride();

        D alpha_d, beta_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));
        __builtin_memcpy(&beta_d, &beta, sizeof(D));

        const bool alpha_zero = (alpha == T(0));
        const bool beta_zero = (beta == T(0));

        const int seg = m;                    // out_len
        constexpr int wlanes = W;
        const int red_len = n;
        const int64_t nbatch = batch;
        const int sgs = sgs_per_wg;

        h.parallel_for<GemvSegNKernel<T, W>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
                const auto sg = it.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int64_t b =
                    static_cast<int64_t>(it.get_group_linear_id()) * sgs +
                    static_cast<int64_t>(sg.get_group_linear_id());

                // SUB-GROUP UNIFORM: the fold must never see a partial sub-group.
                if (b >= nbatch) return;

                const int i = lane % seg;
                const int jsub = lane / seg;

                const D* Ab = a_ptr + b * stride_a;
                const D* xb = x_ptr + b * stride_x;
                D* yb = y_ptr + b * stride_y;

                D acc{};
                if (!alpha_zero && jsub < wlanes) {
                    for (int j = jsub; j < red_len; j += wlanes) {
                        fma_acc(acc, Ab[i + static_cast<int64_t>(j) * lda],
                                xb[static_cast<int64_t>(j) * xinc]);
                    }
                }

                if constexpr (dev_is_complex_v<D>) {
                    auto re = acc.re, im = acc.im;
                    for (int w = wlanes >> 1; w >= 1; w >>= 1) {
                        const int off = seg * w;
                        re += sycl::shift_group_left(sg, re, off);
                        im += sycl::shift_group_left(sg, im, off);
                    }
                    acc = D{re, im};
                } else {
                    for (int w = wlanes >> 1; w >= 1; w >>= 1) {
                        acc += sycl::shift_group_left(sg, acc, seg * w);
                    }
                }

                if (lane < seg) {
                    D out{};
                    fma_acc(out, alpha_d, acc);
                    if (!beta_zero) fma_acc(out, beta_d, yb[static_cast<int64_t>(i) * yinc]);
                    yb[static_cast<int64_t>(i) * yinc] = out;
                }
            });
    });

    return ctx.get_event();
}

// BODY 2 -- {Native, Direct}, transA != NoTrans.  THE PORTABLE ARM.
//   y_j = alpha * sum_i conj?(A[i + j*ld]) * x[i*xinc] + beta * y[j*yinc]
// One work-item per output column. Lanes read `ld` apart and are not coalesced on
// a GPU -- body 3's reason to exist -- but on a device with no 32-lane sub-group
// this is the right shape.
template <typename T>
Event gemv_direct_trans(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        const VectorView<T>& X,
                        const VectorView<T>& Y,
                        T alpha, T beta, bool conjugate) {
    using D = typename DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    const int m = A.rows();
    const int n = A.cols();
    const int batch = A.batch_size();

    const int64_t items = static_cast<int64_t>(n) * batch;

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    const int wg = gemv_wg_ladder(items, max_wg, cu, /*units_per_wg_shift=*/0);
    const int64_t groups = (items + wg - 1) / wg;

    ctx->submit([&](sycl::handler& h) {
        const D* a_ptr = reinterpret_cast<const D*>(A.data_ptr());
        const D* x_ptr = reinterpret_cast<const D*>(X.data_ptr());
        D* y_ptr = reinterpret_cast<D*>(Y.data_ptr());

        const int64_t lda = A.ld();
        const int64_t stride_a = A.stride();
        const int64_t xinc = X.inc();
        const int64_t stride_x = X.stride();
        const int64_t yinc = Y.inc();
        const int64_t stride_y = Y.stride();

        D alpha_d, beta_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));
        __builtin_memcpy(&beta_d, &beta, sizeof(D));

        const bool alpha_zero = (alpha == T(0));
        const bool beta_zero = (beta == T(0));
        const bool conj = conjugate;

        const int out_len = n;
        const int red_len = m;
        const int64_t total = items;

        h.parallel_for<GemvDirectTKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
                if (gid >= total) return;

                const int64_t b = gid / out_len;
                const int j = static_cast<int>(gid - b * out_len);

                const D* Acol = a_ptr + b * stride_a + static_cast<int64_t>(j) * lda;
                const D* xb = x_ptr + b * stride_x;
                D* yb = y_ptr + b * stride_y;

                D acc{};
                if (!alpha_zero) {
                    for (int i = 0; i < red_len; ++i) {
                        D av = Acol[i];
                        // `if constexpr` so a real T emits no branch at all.
                        if constexpr (dev_is_complex_v<D>) {
                            if (conj) av = dev_conj(av);
                        }
                        fma_acc(acc, av, xb[static_cast<int64_t>(i) * xinc]);
                    }
                }

                D out{};
                fma_acc(out, alpha_d, acc);
                if (!beta_zero) fma_acc(out, beta_d, yb[static_cast<int64_t>(j) * yinc]);
                yb[static_cast<int64_t>(j) * yinc] = out;
            });
    });

    return ctx.get_event();
}

// BODY 3 -- {Native, CTA}, transA != NoTrans, GPU with an ENUMERATED sub-group
// size of 32. One 32-lane sub-group per output element: lane l walks the reduction
// as i = l, l+32, ..., reading adjacent elements of one column, then folds through
// sg_sum and lane 0 alone writes y. THE EARLY EXIT IS SUB-GROUP UNIFORM, and has
// to be: a shuffle reached by only some lanes of a sub-group is UB.
template <typename T>
Event gemv_cta_trans(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& A,
                     const VectorView<T>& X,
                     const VectorView<T>& Y,
                     T alpha, T beta, bool conjugate) {
    using D = typename DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    constexpr int kSg = 32;

    const int m = A.rows();
    const int n = A.cols();
    const int batch = A.batch_size();

    const int64_t items = static_cast<int64_t>(n) * batch;

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    const int wg = gemv_wg_ladder(items, max_wg, cu, /*units_per_wg_shift=*/5);
    // supports(CTA) has already established a 32-lane sub-group, so this cannot
    // fire; the alternative to the guard is a division by zero, not a fallback.
    const int sgs_per_wg = (wg / kSg) > 0 ? (wg / kSg) : 1;
    const int64_t groups = (items + sgs_per_wg - 1) / sgs_per_wg;

    ctx->submit([&](sycl::handler& h) {
        const D* a_ptr = reinterpret_cast<const D*>(A.data_ptr());
        const D* x_ptr = reinterpret_cast<const D*>(X.data_ptr());
        D* y_ptr = reinterpret_cast<D*>(Y.data_ptr());

        const int64_t lda = A.ld();
        const int64_t stride_a = A.stride();
        const int64_t xinc = X.inc();
        const int64_t stride_x = X.stride();
        const int64_t yinc = Y.inc();
        const int64_t stride_y = Y.stride();

        D alpha_d, beta_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));
        __builtin_memcpy(&beta_d, &beta, sizeof(D));

        const bool alpha_zero = (alpha == T(0));
        const bool beta_zero = (beta == T(0));
        const bool conj = conjugate;

        const int out_len = n;
        const int red_len = m;
        const int64_t total = items;
        const int sgs = sgs_per_wg;

        h.parallel_for<GemvCtaTKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
                const auto sg = it.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int64_t sg_out =
                    static_cast<int64_t>(it.get_group_linear_id()) * sgs +
                    static_cast<int64_t>(sg.get_group_linear_id());

                // SUB-GROUP UNIFORM: all 32 lanes return here, or none.
                if (sg_out >= total) return;

                const int64_t b = sg_out / out_len;
                const int j = static_cast<int>(sg_out - b * out_len);

                const D* Acol = a_ptr + b * stride_a + static_cast<int64_t>(j) * lda;
                const D* xb = x_ptr + b * stride_x;
                D* yb = y_ptr + b * stride_y;

                D acc{};
                if (!alpha_zero) {
                    for (int i = lane; i < red_len; i += kSg) {
                        D av = Acol[i];
                        if constexpr (dev_is_complex_v<D>) {
                            if (conj) av = dev_conj(av);
                        }
                        fma_acc(acc, av, xb[static_cast<int64_t>(i) * xinc]);
                    }
                }

                acc = sg_sum(sg, acc);

                if (lane == 0) {
                    D out{};
                    fma_acc(out, alpha_d, acc);
                    if (!beta_zero) fma_acc(out, beta_d, yb[static_cast<int64_t>(j) * yinc]);
                    yb[static_cast<int64_t>(j) * yinc] = out;
                }
            });
    });

    return ctx.get_event();
}

// BODY 5 -- GemvSegTKernel<T, W>. {Native, CTA}, transA != NoTrans, GPU with an
// ENUMERATED sub-group size of 32, SHORT REDUCTION. W outputs per sub-group,
// L = 32/W lanes on each -- body 4's mapping TRANSPOSED, not copied:
//
//     BODY 4 (NoTrans)      i = lane % out_len      jsub = lane / out_len
//     BODY 5 (Trans)        s = lane % L            o    = lane / L
//
// Lanes must vary fastest along whichever index is contiguous, which under Trans
// is the REDUCTION index.  evidence: docs/perf/gemv.md#kernel-hypotheses-refuted
//
// THE EARLY EXIT IS **NOT** SUB-GROUP UNIFORM, unlike body 3's: one sub-group
// covers W different outputs, so a tail sub-group is partially in range. It
// returns as a whole only on `base >= total`; the tail is MASKED by `active` and
// sg_out is CLAMPED before any pointer arithmetic. The fold is closed at stride
// 1 (32 % L == 0), so an inactive lane group may safely run it from a zero acc.
template <typename T, int W>
Event gemv_seg_trans(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& A,
                     const VectorView<T>& X,
                     const VectorView<T>& Y,
                     T alpha, T beta, bool conjugate) {
    using D = typename DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");
    static_assert(W >= 2 && W <= 32 && (W & (W - 1)) == 0, "W must be a power of two in [2, 32]");

    constexpr int kSg = 32;
    constexpr int kLanes = kSg / W;          // L -- lanes per output

    const int m = A.rows();
    const int n = A.cols();
    const int batch = A.batch_size();

    const int64_t items = static_cast<int64_t>(n) * batch;
    // W FEWER SUB-GROUPS THAN BODY 3, and the unit handed to the ladder is the
    // SUB-GROUP COUNT: passing `items` here would over-launch by exactly W.
    const int64_t sub_groups = (items + W - 1) / W;

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    const int wg = gemv_wg_ladder(sub_groups, max_wg, cu, /*units_per_wg_shift=*/5);
    const int sgs_per_wg = (wg / kSg) > 0 ? (wg / kSg) : 1;
    const int64_t groups = (sub_groups + sgs_per_wg - 1) / sgs_per_wg;

    ctx->submit([&](sycl::handler& h) {
        // NO __restrict__, for body 1's aliasing reason.
        const D* a_ptr = reinterpret_cast<const D*>(A.data_ptr());
        const D* x_ptr = reinterpret_cast<const D*>(X.data_ptr());
        D* y_ptr = reinterpret_cast<D*>(Y.data_ptr());

        const int64_t lda = A.ld();
        const int64_t stride_a = A.stride();
        const int64_t xinc = X.inc();
        const int64_t stride_x = X.stride();
        const int64_t yinc = Y.inc();
        const int64_t stride_y = Y.stride();

        D alpha_d, beta_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));
        __builtin_memcpy(&beta_d, &beta, sizeof(D));

        const bool alpha_zero = (alpha == T(0));
        const bool beta_zero = (beta == T(0));
        const bool conj = conjugate;

        const int out_len = n;
        const int red_len = m;
        const int64_t total = items;
        const int sgs = sgs_per_wg;

        h.parallel_for<GemvSegTKernel<T, W>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
                const auto sg = it.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int64_t sg_id =
                    static_cast<int64_t>(it.get_group_linear_id()) * sgs +
                    static_cast<int64_t>(sg.get_group_linear_id());
                const int64_t base = sg_id * W;

                // SUB-GROUP UNIFORM: all 32 lanes return here, or none.
                if (base >= total) return;

                const int s = lane % kLanes;         // slice of the reduction
                const int o = lane / kLanes;         // which of the W outputs

                const int64_t sg_out = base + o;
                // MASKED, NOT RETURNED. A tail sub-group is partially in range.
                const bool active = (sg_out < total);
                // CLAMPED, so an inactive group forms no out-of-range address.
                const int64_t sg_out_c = active ? sg_out : (total - 1);

                const int64_t b = sg_out_c / out_len;
                const int j = static_cast<int>(sg_out_c - b * out_len);

                const D* Acol = a_ptr + b * stride_a + static_cast<int64_t>(j) * lda;
                const D* xb = x_ptr + b * stride_x;
                D* yb = y_ptr + b * stride_y;

                D acc{};
                if (!alpha_zero && active) {
                    // kLanes is compile-time, so this loop unrolls -- W's point.
                    for (int i = s; i < red_len; i += kLanes) {
                        D av = Acol[i];
                        if constexpr (dev_is_complex_v<D>) {
                            if (conj) av = dev_conj(av);
                        }
                        fma_acc(acc, av, xb[static_cast<int64_t>(i) * xinc]);
                    }
                }

                if constexpr (dev_is_complex_v<D>) {
                    auto re = acc.re, im = acc.im;
                    for (int off = kLanes >> 1; off >= 1; off >>= 1) {
                        re += sycl::shift_group_left(sg, re, off);
                        im += sycl::shift_group_left(sg, im, off);
                    }
                    acc = D{re, im};
                } else {
                    for (int off = kLanes >> 1; off >= 1; off >>= 1) {
                        acc += sycl::shift_group_left(sg, acc, off);
                    }
                }

                if (s == 0 && active) {
                    D out{};
                    fma_acc(out, alpha_d, acc);
                    if (!beta_zero) fma_acc(out, beta_d, yb[static_cast<int64_t>(j) * yinc]);
                    yb[static_cast<int64_t>(j) * yinc] = out;
                }
            });
    });

    return ctx.get_event();
}

}  // namespace

// TEST-ONLY. Which CTA kernel a (queue, red_len) resolves to: 1 = body 3, W >= 2
// = body 5 at that W. {Native, CTA} names two kernels, so the resolved route
// column cannot tell them apart and a break against body 5 at a shape the gate
// sends to body 3 is vacuous and looks green.
template <typename T>
int gemv_seg_trans_width_debug(Queue& ctx, int red_len, int64_t out_len_times_batch) {
    if constexpr (!kGemvSegTransEmit<T>) {
        static_cast<void>(ctx);
        static_cast<void>(red_len);
        static_cast<void>(out_len_times_batch);
        return 1;
    } else {
        if (!ctx.device().supports_sub_group_size(32)) return 1;
        return gemv_seg_trans_width<T>(
            gemv_segt_mode(), red_len, out_len_times_batch,
            static_cast<int>(ctx.device().get_property(DeviceProperty::MAX_COMPUTE_UNITS)));
    }
}

template <typename T>
Event gemv_native_direct(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const VectorView<T>& X,
                         const VectorView<T>& Y,
                         T alpha, T beta, Transpose transA) {
    if (gemv_quick_return(A.rows(), A.cols(), alpha, beta)) return ctx.get_event();

    if (transA == Transpose::NoTrans) {
        // THE GATE IS THE ENUMERATED SUB-GROUP SIZE, never
        // get_property(MAX_SUB_GROUP_SIZE), which returns sub_group_sizes()[0]:
        // body 4 carries reqd_sub_group_size(32), and a launch without one aborts.
        const int w = gemv_seg_width(A.rows());
        if (w >= 2 && ctx.device().supports_sub_group_size(32)) {
            switch (w) {
                case 32: return gemv_seg_notrans<T, 32>(ctx, A, X, Y, alpha, beta);
                case 16: return gemv_seg_notrans<T, 16>(ctx, A, X, Y, alpha, beta);
                case 8:  return gemv_seg_notrans<T, 8>(ctx, A, X, Y, alpha, beta);
                case 4:  return gemv_seg_notrans<T, 4>(ctx, A, X, Y, alpha, beta);
                default: return gemv_seg_notrans<T, 2>(ctx, A, X, Y, alpha, beta);
            }
        }
        return gemv_direct_notrans<T>(ctx, A, X, Y, alpha, beta);
    }
    return gemv_direct_trans<T>(ctx, A, X, Y, alpha, beta,
                                transA == Transpose::ConjTrans);
}

template <typename T>
Event gemv_native_cta(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const VectorView<T>& X,
                      const VectorView<T>& Y,
                      T alpha, T beta, Transpose transA) {
    if (transA == Transpose::NoTrans) {
        // supports() already gates CTA on transA != NoTrans; reaching here means
        // a caller bypassed the table.
        throw std::runtime_error(
            "BatchLAS: gemv_native_cta called with transA = NoTrans. The CTA body "
            "reduces down a column and serves only Trans/ConjTrans; NoTrans is "
            "already fully coalesced with one work-item per output row and is the "
            "Direct body's job (Algorithm::Direct).");
    }
    if (gemv_quick_return(A.rows(), A.cols(), alpha, beta)) return ctx.get_event();

    // BODY 5 vs BODY 3 -- a device-and-shape choice, not a route, so `native:cta`
    // does not identify which kernel ran (gemv_seg_trans_width_debug does).
    if constexpr (kGemvSegTransEmit<T>) {
        const int w = gemv_seg_trans_width<T>(
            gemv_segt_mode(), A.rows(),
            static_cast<int64_t>(A.cols()) * A.batch_size(),
            static_cast<int>(ctx.device().get_property(DeviceProperty::MAX_COMPUTE_UNITS)));
        if (w >= 2 && ctx.device().supports_sub_group_size(32)) {
            switch (w) {
                case 8: return gemv_seg_trans<T, 8>(ctx, A, X, Y, alpha, beta,
                                                    transA == Transpose::ConjTrans);
                case 4: return gemv_seg_trans<T, 4>(ctx, A, X, Y, alpha, beta,
                                                    transA == Transpose::ConjTrans);
                default: return gemv_seg_trans<T, 2>(ctx, A, X, Y, alpha, beta,
                                                     transA == Transpose::ConjTrans);
            }
        }
    }

    return gemv_cta_trans<T>(ctx, A, X, Y, alpha, beta,
                             transA == Transpose::ConjTrans);
}

// The capability flags live in the same TU as the kernels, so a build that drops
// this file cannot advertise an unlinked kernel. They describe the BUILD, not the
// device -- the Direct route has NO GPU GATE.
template <> bool gemv_direct_available<float>()                { return true; }
template <> bool gemv_direct_available<double>()               { return true; }
template <> bool gemv_direct_available<std::complex<float>>()  { return true; }
template <> bool gemv_direct_available<std::complex<double>>() { return true; }

template <> bool gemv_cta_available<float>()                { return true; }
template <> bool gemv_cta_available<double>()               { return true; }
template <> bool gemv_cta_available<std::complex<float>>()  { return true; }
template <> bool gemv_cta_available<std::complex<double>>() { return true; }

#define BATCHLAS_GEMV_NATIVE_INSTANTIATE(fp)                                   \
    template Event gemv_native_direct<fp>(                                     \
        Queue&, const MatrixView<fp, MatrixFormat::Dense>&,                    \
        const VectorView<fp>&, const VectorView<fp>&, fp, fp, Transpose);      \
    template Event gemv_native_cta<fp>(                                        \
        Queue&, const MatrixView<fp, MatrixFormat::Dense>&,                    \
        const VectorView<fp>&, const VectorView<fp>&, fp, fp, Transpose);      \
    template int gemv_seg_trans_width_debug<fp>(Queue&, int, int64_t);

BATCHLAS_GEMV_NATIVE_INSTANTIATE(float)
BATCHLAS_GEMV_NATIVE_INSTANTIATE(double)
BATCHLAS_GEMV_NATIVE_INSTANTIATE(std::complex<float>)
BATCHLAS_GEMV_NATIVE_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GEMV_NATIVE_INSTANTIATE

}  // namespace batchlas::sycl_gemv
