// SPDX-License-Identifier: MIT
// Native batched GEMV: five kernel bodies, the work-group ladder they share, and
// the capability flags. src/ops/gemv/gemv_native.hh holds the layout premise.
// The bodies are device LA v2 op sequences (partial dot, lane fold, fused epilogue)
// that keep the hand code's fma chain, int64 index arithmetic and shuffle trees.
// evidence: docs/perf/gemv.md#the-five-kernel-bodies

#include <src/ops/gemv/gemv_native.hh>

#include <src/kernels/common/dla_v2_scalar.hh>
#include <src/queue.hh>
#include <src/sycl/device_scalar.hh>
#include <src/sycl/kernel_attrs.hh>
#include <src/util/batch_mask.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <batchlas/blas/device/v2/ext/gemv-native.hh>
#include <batchlas/settings.hh>

namespace batchlas::sycl_gemv {

namespace v2 = batchlas::device::v2;
using sycl_device::DevMap;

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

// One launch's batched operands, re-typed to the device scalar D at the pointer
// boundary. NO __restrict__ ON ANY OF THEM: ortho.cc passes views into the same
// allocation, and __restrict__ is a promise about the object. Item b's views keep
// the hand code's int64 index arithmetic (v2::WideMat / WideVec).
// The alpha == 0 / beta == 0 flags are NOT members: each kernel captures them as
// top-level bools. A bool inside this by-value struct is tested where it is used, so
// beta's flag sat in a register across the dot loop (GemvDirectTKernel<double>: REG
// 36 -> 40 in the sm_89 SASS gate); a top-level bool is an entry-block predicate.
// `beta_zero` is what keeps y unread when beta == 0; y may be uninitialised.
template <typename T>
struct GemvOperands {
    using D = typename DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    const D* a;
    const D* x;
    D* y;
    int64_t lda, stride_a, xinc, stride_x, yinc, stride_y;
    int m, n;
    D alpha, beta;

    GemvOperands(const MatrixView<T, MatrixFormat::Dense>& A, const VectorView<T>& X,
                 const VectorView<T>& Y, T alpha_, T beta_)
        : a(reinterpret_cast<const D*>(A.data_ptr())),
          x(reinterpret_cast<const D*>(X.data_ptr())),
          y(reinterpret_cast<D*>(Y.data_ptr())),
          lda(A.ld()), stride_a(A.stride()), xinc(X.inc()), stride_x(X.stride()),
          yinc(Y.inc()), stride_y(Y.stride()), m(A.rows()), n(A.cols()) {
        __builtin_memcpy(&alpha, &alpha_, sizeof(D));
        __builtin_memcpy(&beta, &beta_, sizeof(D));
    }

    v2::WideMat<const D> mat(int64_t b) const { return v2::wide_mat(a + b * stride_a, m, n, lda); }
    v2::WideVec<const D> xvec(int64_t b, int len) const { return v2::wide_vec(x + b * stride_x, len, xinc); }
    v2::WideVec<D> yvec(int64_t b, int len) const { return v2::wide_vec(y + b * stride_y, len, yinc); }
};

template <typename T> class GemvDirectNKernel;
template <typename T> class GemvDirectTKernel;
template <typename T> class GemvCtaTKernel;
template <typename T, int W> class GemvSegNKernel;
template <typename T, int W> class GemvSegTKernel;

// Body 4's segment width: the largest power of two W with W * out_len <= 32, so
// one sub-group covers a whole output vector; W == 1 gates body 4 off. W must
// stay a TEMPLATE parameter -- a runtime trip count stops the fold unrolling.
// evidence: docs/perf/gemv.md#the-body-4-gate
inline int gemv_seg_width(int out_len) {
    if (out_len <= 0) return 1;
    int w = 1;
    while (w * 2 * out_len <= 32) w *= 2;
    return w;
}

// Body 5's maximum segment width -- W outputs per sub-group, L = 32/W lanes each.
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

// GATE 3 -- a floor on the LAUNCH, per W: body 5 launches W times fewer
// sub-groups than body 3.  evidence: docs/perf/gemv.md#the-body-5-gates
inline int gemv_seg_trans_min_items(int cu, int w) {
    const int c = (cu > 0) ? cu : 1;
    return (w >= 8 ? 16 : 64) * c;
}

// BATCHLAS_GEMV_SEGT = off | auto (default) | 2|4|8 (force body 5 at that W);
// bodies 3 and 5 share a route. Read settings() per call, NEVER latch in a static:
// a latched value makes the A/B measure the default arm twice.
// evidence: docs/perf/gemv.md#the-sub-route-gates
enum class SegTMode { kAuto, kOff, kForce2, kForce4, kForce8 };

inline SegTMode gemv_segt_mode() {
    const char* const s = batchlas::settings().selection.gemv_segt.get();
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

// BODY 1 -- `direct`, transA == NoTrans, one work-item per output row:
//   y_i = alpha * sum_j A[i + j*ld] * x[j*xinc] + beta * y[i*yinc]
// Coalesced ONLY FOR out_len >= 32, since out_len*batch is the only parallel
// extent; body 4 covers shorter outputs where a 32-lane sub-group exists.
template <typename T>
Event gemv_direct_notrans(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& A,
                          const VectorView<T>& X,
                          const VectorView<T>& Y,
                          T alpha, T beta, const std::uint8_t* mask) {
    using D = typename DevMap<T>::type;

    const int m = A.rows();
    const int batch = A.batch_size();

    const int64_t items = static_cast<int64_t>(m) * batch;

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    const int wg = gemv_wg_ladder(items, max_wg, cu, /*units_per_wg_shift=*/0);
    const int64_t groups = (items + wg - 1) / wg;

    ctx->submit([&](sycl::handler& h) {
        const GemvOperands<T> op(A, X, Y, alpha, beta);
        const bool alpha_zero = (alpha == T(0)), beta_zero = (beta == T(0));
        const int64_t total = items;

        batchlas::impl::parallel_for<GemvDirectNKernel<T>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
                if (gid >= total) return;

                const int64_t b = gid / op.m;
                if (mask != nullptr && mask[b] == 0) return;  // I3: an inactive item's y is untouched
                const int i = static_cast<int>(gid - b * op.m);

                // y(i) := alpha A(i, :) x + beta y(i); alpha == 0 skips the dot.
                const auto Ai = op.mat(b).row(i);
                const auto xb = op.xvec(b, op.n);
                const auto yb = op.yvec(b, op.m);
                D acc{};
                if (!alpha_zero) acc = v2::dot_owned_fma(v2::Thread{}, Ai, xb);
                v2::store_fma(v2::rt_scalar(op.alpha, alpha_zero), acc, v2::rt_scalar(op.beta, beta_zero), yb(i));
            });
    });

    return ctx.get_event();
}

// BODY 4 -- `direct`, transA == NoTrans, SHORT OUTPUT. W lanes per
// output, one sub-group per batch item: lane l takes i = l % m, jsub = l / m, so
// lanes l and l+m fold at stride m. THE FOLD IS CLOSED -- lane i draws only from
// lanes i + m*t, t < W, all below m*W <= 32 -- so lanes at or above m*W may shift
// past lane 31 harmlessly; `jsub < W` is a work saving, not a correctness gate.
template <typename T, int W>
Event gemv_seg_notrans(Queue& ctx,
                       const MatrixView<T, MatrixFormat::Dense>& A,
                       const VectorView<T>& X,
                       const VectorView<T>& Y,
                       T alpha, T beta, const std::uint8_t* mask) {
    using D = typename DevMap<T>::type;

    constexpr int kSg = 32;

    const int batch = A.batch_size();

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    const int wg = gemv_wg_ladder(batch, max_wg, cu, /*units_per_wg_shift=*/5);
    const int sgs_per_wg = (wg / kSg) > 0 ? (wg / kSg) : 1;
    const int64_t groups = (static_cast<int64_t>(batch) + sgs_per_wg - 1) / sgs_per_wg;

    ctx->submit([&](sycl::handler& h) {
        const GemvOperands<T> op(A, X, Y, alpha, beta);
        const bool alpha_zero = (alpha == T(0)), beta_zero = (beta == T(0));
        const int64_t nbatch = batch;
        const int sgs = sgs_per_wg;

        batchlas::impl::parallel_for<GemvSegNKernel<T, W>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(kSg) {
                const auto sg = it.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int64_t b =
                    static_cast<int64_t>(it.get_group_linear_id()) * sgs +
                    static_cast<int64_t>(sg.get_group_linear_id());

                // SUB-GROUP UNIFORM: the fold must never see a partial sub-group. b is the
                // sub-group's item, so the mask test is uniform too.
                if (b >= nbatch || (mask != nullptr && mask[b] == 0)) return;

                // Lane = (row i, member jsub of W): member jsub takes j = jsub, jsub + W, ...
                const int seg = op.m;
                const int i = lane % seg;
                const int jsub = lane / seg;

                const auto Ai = op.mat(b).row(i);
                const auto xb = op.xvec(b, op.n);
                const auto yb = op.yvec(b, op.m);
                D acc{};
                if (!alpha_zero && jsub < W) acc = v2::dot_owned_fma(v2::Member<W>{jsub}, Ai, xb);
                acc = v2::fold_down<W>(sg, acc, v2::Dyn{seg});  // lanes i + seg*t onto lane i
                if (lane < seg)
                    v2::store_fma(v2::rt_scalar(op.alpha, alpha_zero), acc, v2::rt_scalar(op.beta, beta_zero), yb(i));
            });
    });

    return ctx.get_event();
}

// BODY 2 -- `direct`, transA != NoTrans.  THE PORTABLE ARM.
//   y_j = alpha * sum_i conj?(A[i + j*ld]) * x[i*xinc] + beta * y[j*yinc]
// Lanes read `ld` apart and are not coalesced; body 3 is the GPU shape.
template <typename T>
Event gemv_direct_trans(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        const VectorView<T>& X,
                        const VectorView<T>& Y,
                        T alpha, T beta, bool conjugate, const std::uint8_t* mask) {
    using D = typename DevMap<T>::type;

    const int n = A.cols();
    const int batch = A.batch_size();

    const int64_t items = static_cast<int64_t>(n) * batch;

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    const int wg = gemv_wg_ladder(items, max_wg, cu, /*units_per_wg_shift=*/0);
    const int64_t groups = (items + wg - 1) / wg;

    ctx->submit([&](sycl::handler& h) {
        const GemvOperands<T> op(A, X, Y, alpha, beta);
        const bool alpha_zero = (alpha == T(0)), beta_zero = (beta == T(0));
        const bool conj = conjugate;
        const int64_t total = items;

        batchlas::impl::parallel_for<GemvDirectTKernel<T>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
                if (gid >= total) return;

                const int64_t b = gid / op.n;
                if (mask != nullptr && mask[b] == 0) return;  // I3
                const int j = static_cast<int>(gid - b * op.n);

                // y(j) := alpha op(A)(j, :) x + beta y(j), op(A)(j, :) = conj?(A(:, j))^T.
                const auto Aj = v2::conj_rt(op.mat(b).col(j), conj);
                const auto xb = op.xvec(b, op.m);
                const auto yb = op.yvec(b, op.n);
                D acc{};
                if (!alpha_zero) acc = v2::dot_owned_fma(v2::Thread{}, Aj, xb);
                v2::store_fma(v2::rt_scalar(op.alpha, alpha_zero), acc, v2::rt_scalar(op.beta, beta_zero), yb(j));
            });
    });

    return ctx.get_event();
}

// BODY 3 -- `cta`, transA != NoTrans, GPU with an ENUMERATED sub-group
// size of 32. One 32-lane sub-group per output element, folded through
// v2::leader_sum, lane 0 alone writes y. THE EARLY EXIT IS SUB-GROUP UNIFORM,
// and has to be: a shuffle reached by only some lanes of a sub-group is UB.
template <typename T>
Event gemv_cta_trans(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& A,
                     const VectorView<T>& X,
                     const VectorView<T>& Y,
                     T alpha, T beta, bool conjugate, const std::uint8_t* mask) {
    using D = typename DevMap<T>::type;

    constexpr int kSg = 32;

    const int n = A.cols();
    const int batch = A.batch_size();

    const int64_t items = static_cast<int64_t>(n) * batch;

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    const int wg = gemv_wg_ladder(items, max_wg, cu, /*units_per_wg_shift=*/5);
    // can_run(cta) has already established a 32-lane sub-group, so this cannot
    // fire; the alternative to the guard is a division by zero, not a fallback.
    const int sgs_per_wg = (wg / kSg) > 0 ? (wg / kSg) : 1;
    const int64_t groups = (items + sgs_per_wg - 1) / sgs_per_wg;

    ctx->submit([&](sycl::handler& h) {
        const GemvOperands<T> op(A, X, Y, alpha, beta);
        const bool alpha_zero = (alpha == T(0)), beta_zero = (beta == T(0));
        const bool conj = conjugate;
        const int64_t total = items;
        const int sgs = sgs_per_wg;

        batchlas::impl::parallel_for<GemvCtaTKernel<T>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(kSg) {
                const v2::SubGroup<kSg> team{it.get_sub_group()};
                const int64_t sg_out =
                    static_cast<int64_t>(it.get_group_linear_id()) * sgs +
                    static_cast<int64_t>(team.sg.get_group_linear_id());

                // SUB-GROUP UNIFORM: all 32 lanes return here, or none.
                if (sg_out >= total) return;

                // b is the sub-group's (one output's) item: this return is uniform as well.
                const int64_t b = sg_out / op.n;
                if (mask != nullptr && mask[b] == 0) return;
                const int j = static_cast<int>(sg_out - b * op.n);

                // The team's partial dots, a shift-down fold (no shared memory, unlike
                // reduce_over_group), and the leader alone writes y(j).
                const auto Aj = v2::conj_rt(op.mat(b).col(j), conj);
                const auto xb = op.xvec(b, op.m);
                const auto yb = op.yvec(b, op.n);
                D acc{};
                if (!alpha_zero) acc = v2::dot_owned_fma(team, Aj, xb);
                acc = v2::leader_sum(team, acc);
                if (team.leader())
                    v2::store_fma(v2::rt_scalar(op.alpha, alpha_zero), acc, v2::rt_scalar(op.beta, beta_zero), yb(j));
            });
    });

    return ctx.get_event();
}

// BODY 5 -- `cta`, transA != NoTrans, GPU with an ENUMERATED sub-group
// size of 32, SHORT REDUCTION. W outputs per sub-group, L = 32/W lanes each:
// s = lane % L, o = lane / L -- body 4's mapping TRANSPOSED, because lanes must
// vary fastest along the contiguous index, which under Trans is the reduction.
// evidence: docs/perf/gemv.md#kernel-hypotheses-refuted
//
// THE EARLY EXIT IS **NOT** SUB-GROUP UNIFORM, unlike body 3's: one sub-group
// covers W outputs, so a tail sub-group is partially in range -- it is MASKED by
// `active` and sg_out is CLAMPED before any pointer arithmetic. The fold is
// closed at stride 1 (32 % L == 0), so an inactive lane group folds a zero acc.
template <typename T, int W>
Event gemv_seg_trans(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& A,
                     const VectorView<T>& X,
                     const VectorView<T>& Y,
                     T alpha, T beta, bool conjugate, const std::uint8_t* mask) {
    using D = typename DevMap<T>::type;
    static_assert(W >= 2 && W <= 32 && (W & (W - 1)) == 0, "W must be a power of two in [2, 32]");

    constexpr int kSg = 32;
    constexpr int kLanes = kSg / W;          // L -- lanes per output

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
        const GemvOperands<T> op(A, X, Y, alpha, beta);
        const bool alpha_zero = (alpha == T(0)), beta_zero = (beta == T(0));
        const bool conj = conjugate;
        const int64_t total = items;
        const int sgs = sgs_per_wg;

        batchlas::impl::parallel_for<GemvSegTKernel<T, W>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(kSg) {
                const auto sg = it.get_sub_group();
                const int64_t sg_id =
                    static_cast<int64_t>(it.get_group_linear_id()) * sgs +
                    static_cast<int64_t>(sg.get_group_linear_id());
                const int64_t base = sg_id * W;

                // SUB-GROUP UNIFORM: all 32 lanes return here, or none.
                if (base >= total) return;

                // W teams of L lanes: team o owns output base + o, its lane team.lid()
                // owns the reduction slice i = team.lid() mod L.
                const v2::Partition<kLanes> team{sg};
                const int o = static_cast<int>(sg.get_local_linear_id()) / kLanes;

                const int64_t sg_out = base + o;
                // MASKED, NOT RETURNED. A tail sub-group is partially in range, and its W
                // outputs can straddle items, so the batch mask folds in here too (I2).
                const bool in_range = (sg_out < total);
                // CLAMPED, so an inactive group forms no out-of-range address.
                const int64_t sg_out_c = in_range ? sg_out : (total - 1);

                const int64_t b = sg_out_c / op.n;
                const int j = static_cast<int>(sg_out_c - b * op.n);
                const bool active = in_range && (mask == nullptr || mask[b] != 0);

                // L is compile-time, so the owned-slice loop unrolls -- W's point.
                const auto Aj = v2::conj_rt(op.mat(b).col(j), conj);
                const auto xb = op.xvec(b, op.m);
                const auto yb = op.yvec(b, op.n);
                D acc{};
                if (!alpha_zero && active) acc = v2::dot_owned_fma(team, Aj, xb);
                acc = v2::leader_sum(team, acc);
                if (team.leader() && active)
                    v2::store_fma(v2::rt_scalar(op.alpha, alpha_zero), acc, v2::rt_scalar(op.beta, beta_zero), yb(j));
            });
    });

    return ctx.get_event();
}

}  // namespace

// TEST-ONLY. Which CTA kernel a (queue, red_len) resolves to: 1 = body 3, W >= 2
// = body 5 at that W. `cta` names two kernels, so the trace or coverage spelling
// column cannot tell them apart and a break against body 5 can look green.
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
    const std::uint8_t* const mask = batchlas::detail::mask_ptr(batchlas::detail::call_mask("gemv", A, X, Y));

    if (transA == Transpose::NoTrans) {
        // THE GATE IS THE ENUMERATED SUB-GROUP SIZE, never MAX_SUB_GROUP_SIZE:
        // body 4 carries reqd_sub_group_size(32), a launch without one aborts.
        const int w = gemv_seg_width(A.rows());
        if (w >= 2 && ctx.device().supports_sub_group_size(32)) {
            switch (w) {
                case 32: return gemv_seg_notrans<T, 32>(ctx, A, X, Y, alpha, beta, mask);
                case 16: return gemv_seg_notrans<T, 16>(ctx, A, X, Y, alpha, beta, mask);
                case 8:  return gemv_seg_notrans<T, 8>(ctx, A, X, Y, alpha, beta, mask);
                case 4:  return gemv_seg_notrans<T, 4>(ctx, A, X, Y, alpha, beta, mask);
                default: return gemv_seg_notrans<T, 2>(ctx, A, X, Y, alpha, beta, mask);
            }
        }
        return gemv_direct_notrans<T>(ctx, A, X, Y, alpha, beta, mask);
    }
    return gemv_direct_trans<T>(ctx, A, X, Y, alpha, beta,
                                transA == Transpose::ConjTrans, mask);
}

template <typename T>
Event gemv_native_cta(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const VectorView<T>& X,
                      const VectorView<T>& Y,
                      T alpha, T beta, Transpose transA) {
    if (transA == Transpose::NoTrans) {
        throw batchlas::unsupported(
            "BatchLAS: gemv_native_cta called with transA = NoTrans. The CTA body "
            "reduces down a column and serves only Trans/ConjTrans; NoTrans is "
            "already fully coalesced with one work-item per output row and is the "
            "`direct` choice's job (BATCHLAS_GEMV_ROUTE=direct).");
    }
    if (gemv_quick_return(A.rows(), A.cols(), alpha, beta)) return ctx.get_event();
    const std::uint8_t* const mask = batchlas::detail::mask_ptr(batchlas::detail::call_mask("gemv", A, X, Y));

    // BODY 5 vs BODY 3 is a device-and-shape choice, not a family: a `cta` pin does
    // not identify which kernel ran (gemv_seg_trans_width_debug does).
    if constexpr (kGemvSegTransEmit<T>) {
        const int w = gemv_seg_trans_width<T>(
            gemv_segt_mode(), A.rows(),
            static_cast<int64_t>(A.cols()) * A.batch_size(),
            static_cast<int>(ctx.device().get_property(DeviceProperty::MAX_COMPUTE_UNITS)));
        if (w >= 2 && ctx.device().supports_sub_group_size(32)) {
            switch (w) {
                case 8: return gemv_seg_trans<T, 8>(ctx, A, X, Y, alpha, beta,
                                                    transA == Transpose::ConjTrans, mask);
                case 4: return gemv_seg_trans<T, 4>(ctx, A, X, Y, alpha, beta,
                                                    transA == Transpose::ConjTrans, mask);
                default: return gemv_seg_trans<T, 2>(ctx, A, X, Y, alpha, beta,
                                                     transA == Transpose::ConjTrans, mask);
            }
        }
    }

    return gemv_cta_trans<T>(ctx, A, X, Y, alpha, beta,
                             transA == Transpose::ConjTrans, mask);
}

// The capability flags live in the same TU as the kernels, so a build that drops
// this file cannot advertise an unlinked kernel. The Direct route has NO GPU GATE.
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
