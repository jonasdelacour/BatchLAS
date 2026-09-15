#pragma once

// Shared device helpers for the register-resident "tiny" factorization tier
// (potrf_tiny.cc, getrf_tiny.cc, geqrf_tiny.cc): one matrix per SubGroupPartition<N>,
// lane r owning row r of a compile-time `D rA[N]`, every cross-lane value a sub-group
// shuffle. THREE INVARIANTS, each breaking SILENTLY: rA[] is never dynamically indexed
// nor a by-reference parameter; N must divide the 32-lane sub-group; every lane of a
// partition must reach every partition collective, so no kernel here returns early.
// evidence: docs/perf/potrf.md#the-shared-tiny-tier-invariants

#include "sg_compat.hh"

#include "../sycl/device_scalar.hh"

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

// COLLECTIVE: `src` must be partition-uniform, every lane must reach this call, and a
// lane guard belongs INSIDE it. Complex shuffles as two reals: permute/select reject an
// aggregate (gesvdj_cta.cc:94).
template <typename D, std::size_t P>
inline D tiny_bcast(SubGroupPartition<P> part, D v, uint32_t src) {
    if constexpr (sycl_device::dev_is_complex_v<D>) {
        return D{select_from_group(part, v.re, src), select_from_group(part, v.im, src)};
    } else {
        return select_from_group(part, v, src);
    }
}

// COMPONENT BY COMPONENT, never a plain `?:` on a complex scalar: LLVM will not build a
// `select` of a 16-byte aggregate, SROA then declines to promote rA[] and the array leaves
// the register file. evidence: docs/perf/potrf.md#the-shared-tiny-tier-invariants

template <typename D>
inline D tiny_select(bool cond, D a, D b) {
    if constexpr (sycl_device::dev_is_complex_v<D>) {
        return D{cond ? a.re : b.re, cond ? a.im : b.im};
    } else {
        return cond ? a : b;
    }
}

// A row that is not `row_live`, and every row of a dead partition, carries the IDENTITY,
// which is what lets these unrolled bodies run with no lane guard. `c <= lane` is
// ortho.cc's OtherTriangleIsNeitherReadNorWritten contract; `c < N` writes into the ld pad.
template <typename D, int N>
inline void tiny_pad_identity(D* rA, int lane) {
    const D one = sycl_device::dev_one<D>();
    const D zero = D{};
#pragma unroll
    for (int c = 0; c < N; ++c) rA[c] = tiny_select(c == lane, one, zero);
}

template <typename D, int N>
inline void tiny_load_lower(D* rA, const D* __restrict a, int ld,
                            int lane, bool row_live,
                            bool real_diag) {  // as LAPACK and cuSOLVER, diag is real
    tiny_pad_identity<D, N>(rA, lane);
#pragma unroll
    for (int c = 0; c < N; ++c) {
        if (row_live && c <= lane) {
            D v = a[lane + static_cast<std::ptrdiff_t>(c) * ld];
            if (real_diag && c == lane) {
                v = sycl_device::dev_from_real<D>(sycl_device::dev_real(v));
            }
            rA[c] = v;
        }
    }
}

// Upper as a LOAD TRANSFORM, not a second algorithm: A = U^H U is the same recurrence on
// S(i,c) = conj(A(c,i)). "Lane r owns a row of U" would need rA[lane] -- invariant 1.
template <typename D, int N>
inline void tiny_load_upper(D* rA, const D* __restrict a, int ld,
                            int lane, bool row_live, bool real_diag) {
    tiny_pad_identity<D, N>(rA, lane);
#pragma unroll
    for (int c = 0; c < N; ++c) {
        if (row_live && c <= lane) {
            D v = sycl_device::dev_conj(a[c + static_cast<std::ptrdiff_t>(lane) * ld]);
            if (real_diag && c == lane) {
                v = sycl_device::dev_from_real<D>(sycl_device::dev_real(v));
            }
            rA[c] = v;
        }
    }
}

template <typename D, int N>
inline void tiny_store_lower(const D* rA, D* __restrict a, int ld,
                             int lane, bool row_live) {
#pragma unroll
    for (int c = 0; c < N; ++c) {
        if (row_live && c <= lane) a[lane + static_cast<std::ptrdiff_t>(c) * ld] = rA[c];
    }
}

template <typename D, int N>
inline void tiny_store_upper(const D* rA, D* __restrict a, int ld,
                             int lane, bool row_live) {
#pragma unroll
    for (int c = 0; c < N; ++c) {
        if (row_live && c <= lane) {
            a[c + static_cast<std::ptrdiff_t>(lane) * ld] = sycl_device::dev_conj(rA[c]);
        }
    }
}

template <typename D, int N>  // the full square, for the non-triangular tiers (LU, QR)
inline void tiny_load_full(D* rA, const D* __restrict a, int ld,
                           int lane, int n, bool row_live) {
    tiny_pad_identity<D, N>(rA, lane);
#pragma unroll
    for (int c = 0; c < N; ++c) {
        if (row_live && c < n) rA[c] = a[lane + static_cast<std::ptrdiff_t>(c) * ld];
    }
}

template <typename D, int N>
inline void tiny_store_full(const D* rA, D* __restrict a, int ld,
                            int row,  // not `lane`: LU relabels rows instead of moving them
                            int n, bool row_live) {
#pragma unroll
    for (int c = 0; c < N; ++c) {
        if (row_live && c < n) a[row + static_cast<std::ptrdiff_t>(c) * ld] = rA[c];
    }
}

// The partition index UNIQUE IN THE WORK-GROUP. The `sg_id *` term is load-bearing
// (steqr_cta.cc:82-86): part.get_group_linear_id() repeats across a work-group's
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

// (magnitude, key) PAIR argmax. The XOR mask is a power of two below N and so uniform
// across the whole 32-lane sub-group, which sg_compat.hh's non-NVPTX fallback requires.
// THE CALLER OWES A NaN-FREE SEED: NaN loses every comparison, so it survives every round
// and the lanes disagree about the winner. evidence: docs/perf/lu.md#pad-rows-and-the-argmax-corrected
template <int N, typename R, typename Part>
inline void tiny_argmax_pair(const Part& part, R& a, int& key) {
#pragma unroll
    for (uint32_t mask = 1u; mask < static_cast<uint32_t>(N); mask <<= 1) {
        const R ov = permute_group_by_xor(part, a, mask);
        const int ok = permute_group_by_xor(part, key, mask);
        if (ov > a || (ov == a && ok < key)) { a = ov; key = ok; }
    }
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
