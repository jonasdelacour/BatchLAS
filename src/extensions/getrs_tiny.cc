// Native batched GETRS, the register-resident tier for order n <= 32 (cdouble n <= 16):
// one SubGroupPartition<N> per matrix, lane r holding row r of op(A) in `D rM[N]` and
// NR right-hand-side columns in `D rB[NR]`. The permutation is a collapsed index each
// lane traces through the interchange list, both substitutions are shuffle
// recurrences, and there is no local memory and no work-group barrier.
// evidence: docs/perf/blackwell.md#lu-getrs-tiny
//
// op(A) = A (NoTrans): rows of A, unit-lower forward then non-unit-upper backward.
// op(A) = A^T / A^H: lane r loads COLUMN r (conjugated for ConjTrans), which makes the
// lower triangle U^T and the upper L^T, so the same body runs with the unit side swapped.
// NoTrans gathers B through the permutation; Trans scatters X through it.
//
// tiny_device.hh's three invariants hold: rM/rB are never dynamically indexed, N divides
// the sub-group, and no lane returns early (a dead partition computes on identity rows).

#include "getrs_native.hh"
#include "tiny_device.hh"

#include "../queue.hh"
#include "../util/resident_capacity.hh"

#include <batchlas/error.hh>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace batchlas {

template <typename T, int N, int NR, bool Trans>
class GetrsTinyKernel;

namespace sycl_getrs {

namespace {

namespace tn = ::batchlas::tiny_native;
namespace sd = ::batchlas::sycl_device;

constexpr int kTinyWg = tn::kTinyWgSize;

// cdouble stops at 16: 32 rows of a 16-byte scalar plus the RHS leave the register file.
template <typename T>
constexpr int getrs_tiny_cap() {
    return std::is_same_v<T, std::complex<double>> ? 16 : 32;
}

// The RHS chunk: wider chunks buy nothing (A stays in registers and every chunk pays the
// same shuffles per column) and cost occupancy -- cfloat n=32 nrhs=16 is 2.39 ms at 16,
// 1.33 at 4, 0.95 at 2. evidence: docs/perf/blackwell.md#lu-getrs-tiny
template <typename T>
constexpr int getrs_tiny_chunk() {
    return sd::dev_is_complex_v<typename sd::DevMap<T>::type> ? 2 : 4;
}

template <typename T>
constexpr int getrs_tiny_rhs_bucket(int nrhs) {
    constexpr int kChunk = getrs_tiny_chunk<T>();
    return nrhs <= 1 ? 1 : nrhs <= 2 ? 2 : kChunk;
}

template <typename D, int N, int NR, bool Trans>
struct GetrsTinyBody {
    const D* ap;
    D* bp;
    const int* piv;
    int n;
    int nrhs;
    int batch;
    bool conj;
    std::ptrdiff_t lda, ldb, stra, strb;

    // y / d for this lane's OWN diagonal: 0 multiplies by the reciprocal, 1 is getrf_tiny's
    // sfmin divide, 2 is a zero pivot, which yields zero. The reciprocal is taken once per
    // lane, not once per step: per-step divides were hoisted out of the RHS chunk loop as
    // N live invariants (255 registers, 17% occupancy on cfloat n=32).
    static D scale(D y, D diag, D dinv, int mode) {
        if (mode == 0) return sd::dev_mul(y, dinv);
        if (mode == 1) return sd::dev_div(y, diag);
        return D{};
    }

    [[sycl::reqd_sub_group_size(32)]]
    void operator()(sycl::nd_item<1> it) const {
        constexpr int kMpw = kTinyWg / N;
        const auto sg = it.get_sub_group();
        const auto part = make_partition<N>(sg);
        const int lane = static_cast<int>(part.get_local_linear_id());
        const int prob_id = static_cast<int>(it.get_group_linear_id()) * kMpw +
                            tn::tiny_partition_id(sg, part);

        const bool live = (prob_id < batch);   // CLAMP, DO NOT RETURN (invariant 3)
        const int b = live ? prob_id : 0;
        const bool row_live = live && lane < n;
        const D* const A = ap + static_cast<std::ptrdiff_t>(b) * stra;
        D* const B = bp + static_cast<std::ptrdiff_t>(b) * strb;

        D rM[N];
#pragma unroll
        for (int c = 0; c < N; ++c) {
            if constexpr (Trans) {
                D v = tn::tiny_load_pad_identity<D>(A, c, lane, n, lda, live);
                if (conj) v = sd::dev_conj(v);
                rM[c] = v;
            } else {
                rM[c] = tn::tiny_load_pad_identity<D>(A, lane, c, n, lda, live);
            }
        }

        // src: the row of the UNPERMUTED vector that position `lane` of F b holds, F the
        // interchange list applied forwards; traced backwards through the list. ?GETRF
        // guarantees p in [j, n); a bad value is clamped so it cannot leave the matrix.
        int myp = lane;
        if (row_live) {
            myp = piv[static_cast<std::ptrdiff_t>(b) * n + lane] - 1;
            if (myp < 0 || myp >= n) myp = lane;
        }
        int src = lane;
#pragma unroll
        for (int j = N - 1; j >= 0; --j) {
            if (j >= n) continue;
            const int p = select_from_group(part, myp, static_cast<uint32_t>(j));
            if (src == j) src = p;
            else if (src == p) src = j;
        }
        const int load_row = Trans ? lane : src;
        const int store_row = Trans ? src : lane;

        D diag = D{};
#pragma unroll
        for (int c = 0; c < N; ++c) diag = tn::tiny_select(c == lane, rM[c], diag);
        const D dinv = sd::dev_recip(diag);
        const int mode = sd::dev_is_zero(diag)                                ? 2
                         : (sd::dev_isfinite(dinv) && !sd::dev_is_zero(dinv)) ? 0
                                                                              : 1;

#pragma unroll 1
        for (int c0 = 0; c0 < nrhs; c0 += NR) {
            // Pad columns are zero and stay finite-or-unstored: nothing reads them back.
            D rB[NR];
#pragma unroll
            for (int k = 0; k < NR; ++k) {
                D v = D{};
                if (row_live && c0 + k < nrhs) {
                    v = B[static_cast<std::ptrdiff_t>(load_row) +
                          static_cast<std::ptrdiff_t>(c0 + k) * ldb];
                }
                rB[k] = v;
            }

            // Forward: lower triangle of op(A); unit for NoTrans (L), not for Trans (U^T).
#pragma unroll
            for (int j = 0; j < N; ++j) {
                if (j >= n) continue;
                const bool upd = lane > j;
#pragma unroll
                for (int k = 0; k < NR; ++k) {
                    D xj;
                    if constexpr (Trans) {
                        xj = tn::tiny_bcast<D>(part, scale(rB[k], diag, dinv, mode),
                                               static_cast<uint32_t>(j));
                        if (lane == j) rB[k] = xj;
                    } else {
                        xj = tn::tiny_bcast<D>(part, rB[k], static_cast<uint32_t>(j));
                    }
                    if (upd) rB[k] = sd::dev_sub(rB[k], sd::dev_mul(rM[j], xj));
                }
            }

            // Backward: upper triangle; non-unit for NoTrans (U), unit for Trans (L^T).
#pragma unroll
            for (int i = N - 1; i >= 0; --i) {
                if (i >= n) continue;
                const bool upd = lane < i;
#pragma unroll
                for (int k = 0; k < NR; ++k) {
                    D xi;
                    if constexpr (Trans) {
                        xi = tn::tiny_bcast<D>(part, rB[k], static_cast<uint32_t>(i));
                    } else {
                        xi = tn::tiny_bcast<D>(part, scale(rB[k], diag, dinv, mode),
                                               static_cast<uint32_t>(i));
                        if (lane == i) rB[k] = xi;
                    }
                    if (upd) rB[k] = sd::dev_sub(rB[k], sd::dev_mul(rM[i], xi));
                }
            }

            // In place and permuted: every row's load must land before any lane stores.
            sycl::group_barrier(sg);
#pragma unroll
            for (int k = 0; k < NR; ++k) {
                if (row_live && c0 + k < nrhs) {
                    B[static_cast<std::ptrdiff_t>(store_row) +
                      static_cast<std::ptrdiff_t>(c0 + k) * ldb] = rB[k];
                }
            }
            sycl::group_barrier(sg);
        }
    }
};

template <typename T, int N, int NR, bool Trans>
Event getrs_tiny_launch(Queue& ctx, const T* a_ptr, int lda, int stride_a, T* b_ptr, int ldb,
                        int stride_b, const int* piv, int n, int nrhs, int batch, bool conj) {
    using D = typename sd::DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");
    static_assert(tn::tiny_n_is_legal(N), "the tiny ladder is {8, 16, 32}");
    using Body = GetrsTinyBody<D, N, NR, Trans>;
    constexpr int kMpw = kTinyWg / N;
    const std::size_t num_wg = static_cast<std::size_t>((batch + kMpw - 1) / kMpw);

    ctx->submit([&](sycl::handler& h) {
        const Body body{reinterpret_cast<const D*>(a_ptr), reinterpret_cast<D*>(b_ptr), piv,
                        n, nrhs, batch, conj,
                        static_cast<std::ptrdiff_t>(lda), static_cast<std::ptrdiff_t>(ldb),
                        static_cast<std::ptrdiff_t>(stride_a),
                        static_cast<std::ptrdiff_t>(stride_b)};
        h.parallel_for<GetrsTinyKernel<T, N, NR, Trans>>(
            sycl::nd_range<1>(sycl::range<1>(num_wg * kTinyWg), sycl::range<1>(kTinyWg)),
            body);
    });
    return ctx.get_event();
}

template <typename T, int N, bool Trans>
Event getrs_tiny_launch_nr(Queue& ctx, const T* a_ptr, int lda, int stride_a, T* b_ptr,
                           int ldb, int stride_b, const int* piv, int n, int nrhs, int batch,
                           bool conj) {
    switch (getrs_tiny_rhs_bucket<T>(nrhs)) {
#define BATCHLAS_GETRS_TINY_NR(RR)                                                       \
        case RR:                                                                         \
            if constexpr (RR <= getrs_tiny_chunk<T>()) {                                  \
                return getrs_tiny_launch<T, N, RR, Trans>(ctx, a_ptr, lda, stride_a,      \
                                                          b_ptr, ldb, stride_b, piv, n,   \
                                                          nrhs, batch, conj);             \
            }                                                                            \
            break;
        BATCHLAS_GETRS_TINY_NR(1) BATCHLAS_GETRS_TINY_NR(2) BATCHLAS_GETRS_TINY_NR(4)
#undef BATCHLAS_GETRS_TINY_NR
        default:
            break;
    }
    throw batchlas::unsupported("getrs_tiny: no RHS bucket for nrhs " + std::to_string(nrhs));
}

template <typename T, bool Trans>
Event getrs_tiny_launch_n(Queue& ctx, const T* a_ptr, int lda, int stride_a, T* b_ptr,
                          int ldb, int stride_b, const int* piv, int n, int nrhs, int batch,
                          bool conj) {
    switch (tn::tiny_bucket_ge(n)) {
        case 8:
            return getrs_tiny_launch_nr<T, 8, Trans>(ctx, a_ptr, lda, stride_a, b_ptr, ldb,
                                                     stride_b, piv, n, nrhs, batch, conj);
        case 16:
            return getrs_tiny_launch_nr<T, 16, Trans>(ctx, a_ptr, lda, stride_a, b_ptr, ldb,
                                                      stride_b, piv, n, nrhs, batch, conj);
        case 32:
            if constexpr (getrs_tiny_cap<T>() >= 32) {
                return getrs_tiny_launch_nr<T, 32, Trans>(ctx, a_ptr, lda, stride_a, b_ptr,
                                                          ldb, stride_b, piv, n, nrhs, batch,
                                                          conj);
            }
            break;
        default:
            break;
    }
    throw batchlas::unsupported("getrs_tiny: order " + std::to_string(n) +
                                " is above this type's register-resident ceiling of " +
                                std::to_string(getrs_tiny_cap<T>()));
}

}  // namespace

template <typename T>
int getrs_tiny_max_n() {
    return getrs_tiny_cap<T>();
}

// Every supports() gate re-applied: this entry point is reachable without the table.
template <typename T>
Event getrs_tiny_dispatch(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& A,
                          const MatrixView<T, MatrixFormat::Dense>& B,
                          Transpose transA,
                          Span<int64_t> pivots) {
    const int n = static_cast<int>(A.rows());
    const int nrhs = static_cast<int>(B.cols());
    const int batch = static_cast<int>(A.batch_size());

    if (n < 1 || nrhs < 1 || batch < 1) {
        throw batchlas::invalid_argument("getrs_tiny: degenerate extents");
    }
    if (A.rows() != A.cols() || B.rows() != A.rows() || B.batch_size() != A.batch_size()) {
        throw batchlas::invalid_argument("getrs_tiny: A must be square and conform with B");
    }
    if (A.is_heterogeneous() || B.is_heterogeneous()) {
        // `if (j >= n) continue` skips a collective: legal only while n is kernel-uniform.
        throw batchlas::invalid_argument("getrs_tiny: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw batchlas::invalid_argument("getrs_tiny: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        throw batchlas::unsupported("getrs_tiny: device does not offer sub-group size 32");
    }
    if (static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE)) < kTinyWg) {
        throw batchlas::unsupported("getrs_tiny: work-group of " + std::to_string(kTinyWg) +
                                    " exceeds this device's maximum");
    }
    if (pivots.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch)) {
        throw batchlas::invalid_argument("getrs_tiny: pivot span is shorter than n * batch");
    }

    // PACKED 1-BASED int32 in the span's first half, as every GPU getrf writes.
    const int* piv = pivots.as_span<int>().data();
    const bool conj = (transA == Transpose::ConjTrans);
    if (transA == Transpose::NoTrans) {
        return getrs_tiny_launch_n<T, false>(ctx, A.data_ptr(), A.ld(), A.stride(),
                                             B.data_ptr(), B.ld(), B.stride(), piv, n, nrhs,
                                             batch, conj);
    }
    return getrs_tiny_launch_n<T, true>(ctx, A.data_ptr(), A.ld(), A.stride(), B.data_ptr(),
                                        B.ld(), B.stride(), piv, n, nrhs, batch, conj);
}

#define BATCHLAS_GETRS_TINY_INSTANTIATE(T)                                                 \
    template int getrs_tiny_max_n<T>();                                                    \
    template Event getrs_tiny_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, \
                                          const MatrixView<T, MatrixFormat::Dense>&,        \
                                          Transpose, Span<int64_t>);

BATCHLAS_GETRS_TINY_INSTANTIATE(float)
BATCHLAS_GETRS_TINY_INSTANTIATE(double)
BATCHLAS_GETRS_TINY_INSTANTIATE(std::complex<float>)
BATCHLAS_GETRS_TINY_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GETRS_TINY_INSTANTIATE

}  // namespace sycl_getrs
}  // namespace batchlas
