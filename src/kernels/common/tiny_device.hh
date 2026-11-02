// SPDX-License-Identifier: MIT
#pragma once

// Shared device helpers for the register-resident "tiny" factorization tier
// (potrf_tiny.cc, getrf_tiny.cc, geqrf_tiny.cc): one matrix per SubGroupPartition<N>,
// lane r owning row r of a compile-time `D rA[N]`, every cross-lane value a sub-group
// shuffle. THREE INVARIANTS, each breaking SILENTLY: rA[] is never dynamically indexed
// nor a by-reference parameter; N must divide the 32-lane sub-group; every lane of a
// partition must reach every partition collective, so no kernel here returns early.
// evidence: docs/perf/potrf.md#the-shared-tiny-tier-invariants

#include <src/kernels/common/sg_partition/sg_partition.hh>
#include <src/kernels/common/dla_v2_scalar.hh>

#include <batchlas/blas/device/v2/ext/elementwise-glue.hh>

#include <src/sycl/device_scalar.hh>
#include <src/sycl/kernel_attrs.hh>

#include <batchlas/backend_config.h>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::tiny_native {

inline constexpr int kTinySubGroupSize = 32;  // every tiny kernel: reqd_sub_group_size(32)
inline constexpr int kTinySubGroups = 2;      // a tuning constant, not a contract
inline constexpr int kTinyWgSize = kTinySubGroups * kTinySubGroupSize;

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

// COLLECTIVE: `src` partition-uniform, every lane reaches the call (a lane guard goes inside).
template <typename D, std::size_t P>
inline D tiny_bcast(SubGroupPartition<P> part, D v, uint32_t src) {
    return ::batchlas::device::v2::bcast(part, v, src);
}

// Component by component (as v2::sel). evidence: docs/perf/potrf.md#the-shared-tiny-tier-invariants
// Kept by hand: forwarding to v2::sel (inline or always_inline, any body) re-types Cx<float> and
// costs complex<float> potrf/posv/gesv tiny +4..26% SASS, REG 162 -> 110 at N = 32.
template <typename D>
inline D tiny_select(bool cond, D a, D b) {
    if constexpr (sycl_device::dev_is_complex_v<D>) {
        return D{cond ? a.re : b.re, cond ? a.im : b.im};
    } else {
        return cond ? a : b;
    }
}

// Dead rows carry the identity, so the unrolled bodies need no lane guard. `c <= lane` is
// ortho.cc's OtherTriangleIsNeitherReadNorWritten contract; `c < N` writes into the ld pad.
template <typename D, int N>
inline void tiny_pad_identity(D* rA, int lane) {
    ::batchlas::device::v2::rowfrag_identity<N>(rA, lane);
}

template <typename D, int N>
inline void tiny_load_lower(D* rA, const D* __restrict a, int ld,
                            int lane, bool row_live,
                            bool real_diag) {  // as LAPACK and cuSOLVER, diag is real
    ::batchlas::device::v2::rowfrag_load_tri<Uplo::Lower, N>(rA, a, ld, lane, row_live, real_diag);
}

// Upper is a LOAD TRANSFORM (S(i,c) = conj(A(c,i))): a row of U per lane needs rA[lane].
template <typename D, int N>
inline void tiny_load_upper(D* rA, const D* __restrict a, int ld,
                            int lane, bool row_live, bool real_diag) {
    ::batchlas::device::v2::rowfrag_load_tri<Uplo::Upper, N>(rA, a, ld, lane, row_live, real_diag);
}

template <typename D, int N>
inline void tiny_store_lower(const D* rA, D* __restrict a, int ld,
                             int lane, bool row_live) {
    ::batchlas::device::v2::rowfrag_store_tri<Uplo::Lower, N>(rA, a, ld, lane, row_live);
}

template <typename D, int N>
inline void tiny_store_upper(const D* rA, D* __restrict a, int ld,
                             int lane, bool row_live) {
    ::batchlas::device::v2::rowfrag_store_tri<Uplo::Upper, N>(rA, a, ld, lane, row_live);
}

template <typename D, int N>  // the full square, for the non-triangular tiers (LU, QR)
inline void tiny_load_full(D* rA, const D* __restrict a, int ld,
                           int lane, int n, bool row_live) {
    ::batchlas::device::v2::rowfrag_load<N>(rA, a, ld, lane, n, row_live);
}

template <typename D, int N>
inline void tiny_store_full(const D* rA, D* __restrict a, int ld,
                            int row,  // not `lane`: LU relabels rows instead of moving them
                            int n, bool row_live) {
    ::batchlas::device::v2::rowfrag_store<N>(rA, a, ld, row, n, row_live);
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

// The partition index UNIQUE IN THE WORK-GROUP: the `sg_id *` term is load-bearing (steqr_cta.cc:83-87),
// part.get_group_linear_id() repeats across sub-groups and half the batch would go untouched.
template <std::size_t P>
inline int tiny_partition_id(const sycl::sub_group& sg, SubGroupPartition<P> part) {
    return static_cast<int>(sg.get_group_linear_id()) *
               static_cast<int>(part.get_group_linear_range()) +
           static_cast<int>(part.get_group_linear_id());
}


// ONE ELEMENT of the padded tile, DELIBERATELY: per row would take the address of the
// caller's register array in the hottest loop. Pads carry the IDENTITY.
template <typename D>
inline D tiny_load_pad_identity(const D* __restrict base, int r, int c, int n,
                                std::ptrdiff_t ld, bool live) {
    return ::batchlas::device::v2::load_or_identity(base, r, c, n, ld, live);
}

// (magnitude, key) argmax; masks < N are uniform over the sub-group (generic backend). THE
// CALLER OWES A NaN-FREE SEED. evidence: docs/perf/lu.md#pad-rows-and-the-argmax-corrected
template <int N, typename R, typename Part>
inline void tiny_argmax_pair(const Part& part, R& a, int& key) {
    ::batchlas::device::v2::argmax_pair<N>(part, a, key);
}

// The ONLY pivot-key encoding: ORDERING field high (lowest wins), winner's LANE low. Every
// ordering field must stay distinct in a partition. evidence: docs/perf/lu.md#the-pivot-key-encoding
inline constexpr int kTinyKeyLaneBits = 8;
inline constexpr int tiny_key(int order_field, int lane) {
    return (order_field << kTinyKeyLaneBits) | lane;
}
inline constexpr int tiny_key_order(int key) { return key >> kTinyKeyLaneBits; }
inline constexpr uint32_t tiny_key_lane(int key) {
    return static_cast<uint32_t>(key) & ((1u << kTinyKeyLaneBits) - 1u);
}

}  // namespace batchlas::tiny_native
