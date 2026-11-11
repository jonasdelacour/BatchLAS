// SPDX-License-Identifier: MIT
#pragma once

// The register-resident tiny tier's helpers: lane r of a SubGroupPartition<N> owns row r of `D rA[N]`.
// THREE INVARIANTS, each breaking SILENTLY: rA[] is never dynamically indexed nor a by-reference
// parameter; N divides the 32-lane sub-group; no kernel returns early before a partition collective.
// evidence: docs/perf/potrf.md#the-shared-tiny-tier-invariants

#include <batchlas/device/detail/sg_partition/sg_partition.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/device/arith.hh>

#include <src/sycl/device_scalar.hh>
#include <src/util/resident_capacity.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

#include <batchlas/backend_config.h>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::tiny_native {

inline constexpr int kTinySubGroupSize = 32;  // every tiny kernel: reqd_sub_group_size(32)

constexpr bool tiny_n_is_legal(int N) {  // invariant 2: N must divide the sub-group
    return N == 8 || N == 16 || N == 32;
}

constexpr int tiny_bucket_ge(int n) {  // order -> bucket; 0 means "above the tier"
    if (n < 1) return 0;
    if (n <= 8) return 8;
    if (n <= 16) return 16;
    if (n <= 32) return 32;
    return 0;
}

// COLLECTIVE: `src` partition-uniform, every lane reaches it, a lane guard goes INSIDE it.
template <typename D, std::size_t P>
inline D tiny_bcast(SubGroupPartition<P> part, D v, uint32_t src) {
    return ::batchlas::device::wrap(part).bcast(v, src);
}

// Component by component, never `?:` on a complex aggregate (SROA would demote rA[]): core select.
template <typename D>
BATCHLAS_DEVICE_INLINE D tiny_select(bool cond, D a, D b) {
    return ::batchlas::device::select(cond, a, b);
}

// A row that is not `row_live`, and every row of a dead partition, carries the IDENTITY,
// which is what lets these unrolled bodies run with no lane guard.
template <typename D, int N>
inline void tiny_pad_identity(D* rA, int lane) {
    const D one = sycl_device::dev_one<D>();
    const D zero = D{};
BATCHLAS_UNROLL_FULL
    for (int c = 0; c < N; ++c) rA[c] = tiny_select(c == lane, one, zero);
}

template <typename D, int N>  // the full square, for the non-triangular tiers (LU, QR)
inline void tiny_load_full(D* rA, const D* __restrict a, int ld,
                           int lane, int n, bool row_live) {
    tiny_pad_identity<D, N>(rA, lane);
BATCHLAS_UNROLL_FULL
    for (int c = 0; c < N; ++c) {
        if (row_live && c < n) rA[c] = a[lane + static_cast<std::ptrdiff_t>(c) * ld];
    }
}

template <typename D, int N>
inline void tiny_store_full(const D* rA, D* __restrict a, int ld,
                            int row,  // not `lane`: LU relabels rows instead of moving them
                            int n, bool row_live) {
BATCHLAS_UNROLL_FULL
    for (int c = 0; c < N; ++c) {
        if (row_live && c < n) a[row + static_cast<std::ptrdiff_t>(c) * ld] = rA[c];
    }
}

// rA[] is sized by a compile-time NC >= n, not N. evidence: docs/perf/lu.md#the-column-bucket
constexpr int tiny_col_bucket(int n, int step) {
    return ((n + step - 1) / step) * step;
}

// A row broadcast through local memory as 16-byte vectors, packed by value and never by a
// `?:` over the complex aggregate. evidence: docs/perf/lu.md#the-local-memory-broadcast
#if BATCHLAS_DEVICE_NO_VECTOR_TYPES
template <typename R>
struct alignas(4 * sizeof(R)) tiny_r4 {  // Metal: no vector types (src/ops/gemm/kernels/vec4.hh)
    R x, y, z, w;
};
#else
template <typename R>
using tiny_r4 = R __attribute__((ext_vector_type(4)));
#endif

template <typename D>
inline constexpr int tiny_r4_width() {  // D elements per 16-byte vector
    return sycl_device::dev_is_complex_v<D> ? 2 : 4;
}

// Element `e` of a vector; `e` is compile-time once the caller is unrolled.
template <typename D, typename R>
inline D tiny_r4_get(tiny_r4<R> x, int e) {
    if constexpr (sycl_device::dev_is_complex_v<D>) {
        return D{e == 0 ? x.x : x.z, e == 0 ? x.y : x.w};
    } else {
        return e == 0 ? x.x : e == 1 ? x.y : e == 2 ? x.z : x.w;
    }
}

// By VALUE, never a pointer into rA[] (invariant 1); a complex vector takes a and b only.
template <typename D, typename R>
inline tiny_r4<R> tiny_r4_pack(D a, D b, D c, D d) {
    if constexpr (sycl_device::dev_is_complex_v<D>) {
        return tiny_r4<R>{a.re, a.im, b.re, b.im};
    } else {
        return tiny_r4<R>{a, b, c, d};
    }
}

// Vector `v` of the row rA[0..NC). `% NC` only keeps an index in range past a row end;
// those vector lanes are never read. A MACRO because a function would take rA's address.
#define BATCHLAS_TINY_R4_PACK(D, R, NC, rA, v)                                              \
    ::batchlas::tiny_native::tiny_r4_pack<D, R>(                                            \
        rA[((v) * ::batchlas::tiny_native::tiny_r4_width<D>()) % (NC)],                     \
        rA[((v) * ::batchlas::tiny_native::tiny_r4_width<D>() + 1) % (NC)],                 \
        rA[((v) * ::batchlas::tiny_native::tiny_r4_width<D>() + 2) % (NC)],                 \
        rA[((v) * ::batchlas::tiny_native::tiny_r4_width<D>() + 3) % (NC)])

// The partition index UNIQUE IN THE WORK-GROUP. The `sg_id *` term is load-bearing
// (steqr_cta.cc:83-87): part.get_group_linear_id() repeats across a work-group's
// sub-groups, so without it half the batch is never touched. Armed in all three tiers.
template <std::size_t P>
inline int tiny_partition_id(const sycl::sub_group& sg, SubGroupPartition<P> part) {
    return static_cast<int>(sg.get_group_linear_id()) *
               static_cast<int>(part.get_group_linear_range()) +
           static_cast<int>(part.get_group_linear_id());
}

// ONE ELEMENT of the N x N padded tile. Per element, not per row, DELIBERATELY: the one
// spelling that cannot take the address of the caller's register array, in the tier's
// hottest loop. Pad rows and columns carry the IDENTITY; `live` short-circuits.
template <typename D>
inline D tiny_load_pad_identity(const D* __restrict base, int r, int c, int n,
                                std::ptrdiff_t ld, bool live) {
    if (live && r < n && c < n) {
        return base[static_cast<std::ptrdiff_t>(r) + static_cast<std::ptrdiff_t>(c) * ld];
    }
    return (r == c) ? sycl_device::dev_one<D>() : D{};
}

}  // namespace batchlas::tiny_native
