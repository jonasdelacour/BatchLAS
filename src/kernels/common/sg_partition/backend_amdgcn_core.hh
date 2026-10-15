#pragma once
// AMDGCN cross-lane primitives for chunked sub-group partitions, SYCL-free:
// plain clang AMDGPU builtins over 32-bit words, so a HIP TU can compile this
// header to ISA without a DPC++ AMD toolchain. backend_amdgcn.hh adapts it.
//
// Masking is free: divergence is EXEC masking of one instruction stream, and
// every primitive reads only lanes of the caller's chunk, which the contract
// keeps active. (ds_bpermute reads 0 and DPP reads out-of-bounds from a
// disabled lane; only the shifts' unspecified out-of-chunk lanes can see it.)
// The enc:: encodings are builtin-free so a host model can check them.

#include <cstdint>

namespace batchlas::sgp::amdgcn {

// kWave bounds P. Clang dropped __AMDGCN_WAVEFRONT_SIZE__ (intel/llvm defines
// none) and -mwavefrontsize64 leaves no macro, so gfx9 is 64 (no wave32) and
// RDNA assumes 32; RDNA wave64 builds wanting P == 64 define
// BATCHLAS_AMDGCN_WAVE_SIZE=64. P <= 32 is right in both modes: code that
// depends on the real wave size asks wave_size().
#if defined(BATCHLAS_AMDGCN_WAVE_SIZE)
inline constexpr uint32_t kWave = BATCHLAS_AMDGCN_WAVE_SIZE;
inline constexpr bool kWaveExact = true;
#elif defined(__AMDGCN_WAVEFRONT_SIZE__)
inline constexpr uint32_t kWave = __AMDGCN_WAVEFRONT_SIZE__;
inline constexpr bool kWaveExact = true;
#elif defined(__AMDGCN_WAVEFRONT_SIZE)
inline constexpr uint32_t kWave = __AMDGCN_WAVEFRONT_SIZE;
inline constexpr bool kWaveExact = true;
#elif defined(__GFX9__) || !defined(__AMDGCN__)
inline constexpr uint32_t kWave = 64;
inline constexpr bool kWaveExact = true;
#else
inline constexpr uint32_t kWave = 32;
inline constexpr bool kWaveExact = false;
#define BATCHLAS_AMDGCN_WAVE_GUESSED 1
#endif

// DPP16 (row_share/row_xmask) and v_permlane[x]16 arrived with gfx10, which
// dropped wave_shl/wave_shr; v_permlane64 is gfx11+.
#if defined(__GFX12__)
inline constexpr int kGen = 12;
#elif defined(__GFX11__)
inline constexpr int kGen = 11;
#elif defined(__GFX10__)
inline constexpr int kGen = 10;
#elif defined(__GFX9__) || !defined(__AMDGCN__)
inline constexpr int kGen = 9;
#else
#error "sg_partition/amdgcn: only gfx9 (CDNA/Vega) and gfx10+ (RDNA) are supported"
#endif

// gfx90a+ CDNA decodes the row_share encoding as row_newbcast (same meaning).
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || \
    defined(__gfx950__)
inline constexpr bool kRowNewBcast = true;
#else
inline constexpr bool kRowNewBcast = false;
#endif

namespace enc {

// DPP controls (LLVM SIDefines.h DppCtrl); i is the lane, rows are 16 lanes.
inline constexpr int kRowShl0 = 0x100;        // src[i + n], same row
inline constexpr int kRowShr0 = 0x110;        // src[i - n], same row
inline constexpr int kRowRor0 = 0x120;        // src[(i - n) mod 16]
inline constexpr int kWaveShl1 = 0x130;       // gfx9: src[i + 1], whole wave
inline constexpr int kWaveShr1 = 0x138;       // gfx9: src[i - 1], whole wave
inline constexpr int kRowMirror = 0x140;      // src[i ^ 15]
inline constexpr int kRowHalfMirror = 0x141;  // src[i ^ 7]
inline constexpr int kRowShare0 = 0x150;      // src[row + n]
inline constexpr int kRowXmask0 = 0x160;      // gfx10+: src[i ^ n]

constexpr int quad_perm(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
    return static_cast<int>(s0 | (s1 << 2) | (s2 << 4) | (s3 << 6));
}
constexpr int quad_xor(uint32_t m) { return quad_perm(0 ^ m, 1 ^ m, 2 ^ m, 3 ^ m); }
constexpr int quad_bcast(uint32_t P, uint32_t s) {
    const uint32_t hi = ~(P - 1u) & 3u;
    return quad_perm((0 & hi) | s, (1 & hi) | s, (2 & hi) | s, (3 & hi) | s);
}

// v_permlanex16: lane j of a row reads lane sel(j) of the other row of its
// 32-lane half; sel(j) is nibble j of lo (j < 8) or hi (j >= 8).
struct Sel16 {
    uint32_t lo, hi;
};
template <typename F>
constexpr Sel16 sel16(F f) {
    Sel16 s{0, 0};
    for (uint32_t j = 0; j < 8; ++j) s.lo |= (f(j) & 15u) << (4 * j);
    for (uint32_t j = 8; j < 16; ++j) s.hi |= (f(j) & 15u) << (4 * (j - 8));
    return s;
}
constexpr Sel16 sel16_xor(uint32_t m) { return sel16([m](uint32_t j) { return j ^ m; }); }
constexpr Sel16 sel16_add(uint32_t d) { return sel16([d](uint32_t j) { return j + d; }); }
constexpr Sel16 sel16_sub(uint32_t d) { return sel16([d](uint32_t j) { return j - d; }); }

// ds_swizzle bitmask mode within 32 lanes: src = ((i & and) | or) ^ xor.
constexpr int swizzle_xor(uint32_t m) { return static_cast<int>(((m & 31u) << 10) | 0x1Fu); }

// gfx9 has no row_xmask: i ^ m (m < 16) is a mirror (i^7, i^15) or ror 8 for
// the high bits, then a quad_perm for the rest. second == -1: no second move.
struct Gfx9XorPlan {
    int first;
    int second;
};
constexpr Gfx9XorPlan gfx9_xor_plan(uint32_t m) {
    const uint32_t lo = m & 3u, hi = m & 12u;
    if (hi == 0) return {quad_xor(lo), -1};
    if (hi == 8) return {kRowRor0 + 8, lo ? quad_xor(lo) : -1};
    const uint32_t rest = lo ^ 3u;
    return {hi == 4 ? kRowHalfMirror : kRowMirror, rest ? quad_xor(rest) : -1};
}

// Reduction steps in a row. Once quads are uniform, i^7 reaches the partner
// quad as i^4 would, and once 8-lane halves are, i^15 stands in for i^8: one
// DPP move per step where an exact i^4 costs gfx9 two.
inline constexpr int kReduceCtrl[4] = {quad_xor(1), quad_xor(2), kRowHalfMirror, kRowMirror};

}  // namespace enc

#if defined(__AMDGCN__)

#if defined(__HIP__)  // the HIP verification TU needs __device__
#define BATCHLAS_AMDGCN_FN __device__ __attribute__((always_inline)) inline
#else
#define BATCHLAS_AMDGCN_FN __attribute__((always_inline)) inline
#endif

#if defined(BATCHLAS_AMDGCN_WAVE_GUESSED) && !__has_builtin(__builtin_amdgcn_wavefrontsize)
#error "sg_partition/amdgcn: wave size unknown; define BATCHLAS_AMDGCN_WAVE_SIZE to 32 or 64"
#endif

// The backend folds the builtin to the subtarget's constant, so branching on
// it is free.
BATCHLAS_AMDGCN_FN uint32_t wave_size() {
#if defined(BATCHLAS_AMDGCN_WAVE_GUESSED)
    return __builtin_amdgcn_wavefrontsize();
#else
    return kWave;
#endif
}

// A whole-wave chunk may read one lane with v_readlane: scalar, EXEC-blind.
template <uint32_t P>
BATCHLAS_AMDGCN_FN bool whole_wave() {
    if constexpr (P != 32 && P != 64) return false;
    else return P == wave_size();
}

BATCHLAS_AMDGCN_FN uint32_t lane_id() {
    uint32_t l = __builtin_amdgcn_mbcnt_lo(~0u, 0u);
    if constexpr (kWave == 64 || !kWaveExact) l = __builtin_amdgcn_mbcnt_hi(~0u, l);  // +0 in wave32
    return l;
}

// Full masks, bound_ctrl and an undef old let GCNDPPCombine fold the move
// into its consumer (v_add_f32_dpp).
template <int Ctrl>
BATCHLAS_AMDGCN_FN uint32_t dpp(uint32_t v) {
    return static_cast<uint32_t>(__builtin_amdgcn_mov_dpp(static_cast<int>(v), Ctrl, 0xF, 0xF, true));
}

// Out-of-row lanes keep `old`.
template <int Ctrl>
BATCHLAS_AMDGCN_FN uint32_t dpp_or(uint32_t old, uint32_t v) {
    return static_cast<uint32_t>(
        __builtin_amdgcn_update_dpp(static_cast<int>(old), static_cast<int>(v), Ctrl, 0xF, 0xF, false));
}

BATCHLAS_AMDGCN_FN uint32_t bpermute(uint32_t src_lane, uint32_t v) {
    return static_cast<uint32_t>(__builtin_amdgcn_ds_bpermute(static_cast<int>(src_lane << 2), static_cast<int>(v)));
}

BATCHLAS_AMDGCN_FN uint32_t readlane(uint32_t v, uint32_t lane) {
    return static_cast<uint32_t>(__builtin_amdgcn_readlane(static_cast<int>(v), static_cast<int>(lane)));
}

#if !defined(__GFX9__)
template <uint32_t Lo, uint32_t Hi>
BATCHLAS_AMDGCN_FN uint32_t permlanex16(uint32_t old, uint32_t v) {
    return __builtin_amdgcn_permlanex16(old, v, Lo, Hi, false, false);
}
#endif

// v[i ^ 32] for RDNA wave64. gfx10 has no cross-half move at all, so it goes
// lane by lane through SGPRs (~160 instructions, P == 64 only).
template <int D = 0>
BATCHLAS_AMDGCN_FN uint32_t swap_halves(uint32_t v) {
    static_assert(D == 0 && kWave == 64 && kGen >= 10, "swap_halves is an RDNA wave64 operation");
#if defined(__GFX11__) || defined(__GFX12__)
    return __builtin_amdgcn_permlane64(v);
#else
    const uint32_t lane = lane_id();
    uint32_t r = v;
#pragma unroll
    for (uint32_t j = 0; j < 32; ++j) {
        const uint32_t lo = readlane(v, j), hi = readlane(v, j + 32);
        r = (lane & 31u) == j ? ((lane & 32u) ? lo : hi) : r;
    }
    return r;
#endif
}

// Gather from any lane of the chunk. RDNA wave64 ds_bpermute is not
// documented to cross 32-lane halves, so P == 64 reads its own half of v or
// of the swapped v: right whichever way the hardware wraps.
template <uint32_t P>
BATCHLAS_AMDGCN_FN uint32_t gather(uint32_t lane, uint32_t src, uint32_t v) {
    if constexpr (P == 64 && kGen >= 10) {
        const uint32_t addr = (lane & 32u) | (src & 31u);
        const uint32_t same = bpermute(addr, v);
        const uint32_t other = bpermute(addr, swap_halves(v));
        return ((src ^ lane) & 32u) ? other : same;
    } else {
        return bpermute(src, v);
    }
}

template <uint32_t M>
BATCHLAS_AMDGCN_FN uint32_t xor_const(uint32_t lane, uint32_t v) {
    static_assert(M < kWave, "xor mask outside the wave");
    if constexpr (M == 0) {
        return v;
    } else if constexpr (M < 4) {
        return dpp<enc::quad_xor(M)>(v);
    } else if constexpr (M < 16) {
        if constexpr (kGen >= 10) {
            return dpp<enc::kRowXmask0 + static_cast<int>(M)>(v);
        } else {
            constexpr enc::Gfx9XorPlan plan = enc::gfx9_xor_plan(M);
            const uint32_t t = dpp<plan.first>(v);
            if constexpr (plan.second >= 0) return dpp<plan.second>(t);
            else return t;
        }
    } else if constexpr (M < 32) {
#if defined(__GFX9__)
        return static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int>(v), enc::swizzle_xor(M)));
#else
        constexpr enc::Sel16 s = enc::sel16_xor(M & 15u);
        return permlanex16<s.lo, s.hi>(v, v);
#endif
    } else if constexpr (kGen >= 10) {
        return xor_const<M & 31u>(lane, swap_halves(v));
    } else {
        return bpermute(lane ^ M, v);
    }
}

// Folds to one xor_const once inlining makes `mask` a constant.
template <uint32_t P, uint32_t M = 1>
BATCHLAS_AMDGCN_FN uint32_t xor_dispatch(uint32_t lane, uint32_t v, uint32_t mask) {
    if constexpr (M >= P) {
        return gather<P>(lane, lane ^ mask, v);
    } else {
        if (mask == M) return xor_const<M>(lane, v);
        return xor_dispatch<P, M + 1>(lane, v, mask);
    }
}

// A runtime argument goes straight to the crossbar rather than through a
// chain of compares; __builtin_constant_p resolves after inlining.
template <uint32_t P>
BATCHLAS_AMDGCN_FN uint32_t shfl_xor(uint32_t lane, uint32_t v, uint32_t mask) {
    if constexpr (P == 1) {
        return v;
    } else {
        if (__builtin_constant_p(mask)) return xor_dispatch<P>(lane, v, mask);
        return gather<P>(lane, lane ^ mask, v);
    }
}

template <uint32_t P, uint32_t S>
BATCHLAS_AMDGCN_FN uint32_t idx_const(uint32_t lane, uint32_t base, uint32_t v) {
    if (whole_wave<P>()) return readlane(v, S);
    if constexpr (P <= 4) {
        return dpp<enc::quad_bcast(P, S)>(v);
    } else if constexpr (P == 16 && (kGen >= 10 || kRowNewBcast)) {
        return dpp<enc::kRowShare0 + static_cast<int>(S)>(v);
    } else {
        return gather<P>(lane, base + S, v);
    }
}

template <uint32_t P, uint32_t S = 0>
BATCHLAS_AMDGCN_FN uint32_t idx_dispatch(uint32_t lane, uint32_t base, uint32_t v, uint32_t src) {
    if constexpr (S >= P) {
        return gather<P>(lane, base + src, v);
    } else {
        if (src == S) return idx_const<P, S>(lane, base, v);
        return idx_dispatch<P, S + 1>(lane, base, v, src);
    }
}

template <uint32_t P>
BATCHLAS_AMDGCN_FN uint32_t shfl_idx(uint32_t lane, uint32_t base, uint32_t v, uint32_t src) {
    if constexpr (P == 1) {
        return v;
    } else {
        if (__builtin_constant_p(src)) return idx_dispatch<P>(lane, base, v, src);
        return gather<P>(lane, base + src, v);
    }
}

// Out-of-chunk sources are unspecified, so a row-local shift serves P <= 16.
// For P == 32 on gfx10+, lanes whose source is in the other row keep `old`,
// which permlanex16 filled from exactly there.
template <uint32_t P, uint32_t D>
BATCHLAS_AMDGCN_FN uint32_t down_const(uint32_t lane, uint32_t v) {
    if constexpr (P <= 16) {
        return dpp<enc::kRowShl0 + static_cast<int>(D)>(v);
    } else if constexpr (kGen == 9 && D == 1) {
        return dpp<enc::kWaveShl1>(v);
    } else if constexpr (kGen >= 10 && P == 32 && D < 16) {
        constexpr enc::Sel16 s = enc::sel16_add(D);
        return dpp_or<enc::kRowShl0 + static_cast<int>(D)>(permlanex16<s.lo, s.hi>(v, v), v);
    } else {
        return gather<P>(lane, lane + D, v);
    }
}

template <uint32_t P, uint32_t D>
BATCHLAS_AMDGCN_FN uint32_t up_const(uint32_t lane, uint32_t v) {
    if constexpr (P <= 16) {
        return dpp<enc::kRowShr0 + static_cast<int>(D)>(v);
    } else if constexpr (kGen == 9 && D == 1) {
        return dpp<enc::kWaveShr1>(v);
    } else if constexpr (kGen >= 10 && P == 32 && D < 16) {
        constexpr enc::Sel16 s = enc::sel16_sub(D);
        return dpp_or<enc::kRowShr0 + static_cast<int>(D)>(permlanex16<s.lo, s.hi>(v, v), v);
    } else {
        return gather<P>(lane, lane - D, v);
    }
}

template <uint32_t P, uint32_t D = 1>
BATCHLAS_AMDGCN_FN uint32_t down_dispatch(uint32_t lane, uint32_t v, uint32_t delta) {
    if constexpr (D >= P) {
        return delta == 0 ? v : gather<P>(lane, lane + delta, v);
    } else {
        if (delta == D) return down_const<P, D>(lane, v);
        return down_dispatch<P, D + 1>(lane, v, delta);
    }
}

template <uint32_t P, uint32_t D = 1>
BATCHLAS_AMDGCN_FN uint32_t up_dispatch(uint32_t lane, uint32_t v, uint32_t delta) {
    if constexpr (D >= P) {
        return delta == 0 ? v : gather<P>(lane, lane - delta, v);
    } else {
        if (delta == D) return up_const<P, D>(lane, v);
        return up_dispatch<P, D + 1>(lane, v, delta);
    }
}

template <uint32_t P>
BATCHLAS_AMDGCN_FN uint32_t shfl_down(uint32_t lane, uint32_t v, uint32_t delta) {
    if constexpr (P == 1) {
        return v;
    } else {
        if (__builtin_constant_p(delta)) return down_dispatch<P>(lane, v, delta);
        return gather<P>(lane, lane + delta, v);
    }
}

template <uint32_t P>
BATCHLAS_AMDGCN_FN uint32_t shfl_up(uint32_t lane, uint32_t v, uint32_t delta) {
    if constexpr (P == 1) {
        return v;
    } else {
        if (__builtin_constant_p(delta)) return up_dispatch<P>(lane, v, delta);
        return gather<P>(lane, lane - delta, v);
    }
}

// Only active lanes vote, and the chunk is all active. ballot_w64 is the one
// spelling both wave modes accept (wave32 zero-extends).
template <uint32_t P>
BATCHLAS_AMDGCN_FN uint64_t ballot(uint32_t base, bool pred) {
    const uint64_t bits = __builtin_amdgcn_ballot_w64(pred);
    if (whole_wave<P>()) return bits;
    if constexpr (P < 64) return (bits >> base) & ((uint64_t{1} << P) - 1u);
    else return bits;
}

// A wave's memory operations issue in order, so a wavefront-scope fence emits
// no s_waitcnt; it only pins the compiler. Newer clang made
// __builtin_amdgcn_fence variadic, which SYCL device code may not call.
BATCHLAS_AMDGCN_FN void barrier() {
#if __has_builtin(__scoped_atomic_thread_fence)
    __scoped_atomic_thread_fence(__ATOMIC_ACQ_REL, __MEMORY_SCOPE_WVFRNT);
#else
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
#endif
    __builtin_amdgcn_wave_barrier();
}

template <typename T, typename F>
BATCHLAS_AMDGCN_FN T per_word(const T& v, F&& f) {
    static_assert(sizeof(T) % 4 == 0, "reduce moves whole 32-bit words");
    constexpr uint32_t n = sizeof(T) / 4;
    uint32_t w[n];
    __builtin_memcpy(w, &v, sizeof(T));
#pragma unroll
    for (uint32_t i = 0; i < n; ++i) w[i] = f(w[i]);
    T out;
    __builtin_memcpy(&out, w, sizeof(T));
    return out;
}

template <int Ctrl, typename T>
BATCHLAS_AMDGCN_FN T dpp_t(const T& v) {
    return per_word(v, [](uint32_t w) { return dpp<Ctrl>(w); });
}

template <uint32_t Lane, typename T>
BATCHLAS_AMDGCN_FN T readlane_t(const T& v) {
    return per_word(v, [](uint32_t w) { return readlane(w, Lane); });
}

// All-lanes reduction for a commutative, associative Op. A whole-wave chunk
// combines its row results in SGPRs (bitwise-uniform, no crossbar); a 32-lane
// chunk of a wave64 swaps rows by ds_swizzle (gfx9) or permlanex16.
template <uint32_t P, typename T, typename Op>
BATCHLAS_AMDGCN_FN T reduce(T v, Op op) {
    static_assert(P <= kWave, "partition wider than the wave");
    if constexpr (P >= 2) v = op(v, dpp_t<enc::kReduceCtrl[0]>(v));
    if constexpr (P >= 4) v = op(v, dpp_t<enc::kReduceCtrl[1]>(v));
    if constexpr (P >= 8) v = op(v, dpp_t<enc::kReduceCtrl[2]>(v));
    if constexpr (P >= 16) v = op(v, dpp_t<enc::kReduceCtrl[3]>(v));
    if constexpr (P == 64) {
        return op(op(readlane_t<0>(v), readlane_t<16>(v)), op(readlane_t<32>(v), readlane_t<48>(v)));
    } else if constexpr (P == 32) {
        if (whole_wave<32>()) return op(readlane_t<0>(v), readlane_t<16>(v));
#if defined(__GFX9__)
        return op(v, per_word(v, [](uint32_t w) {
                      return static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int>(w), enc::swizzle_xor(16)));
                  }));
#else
        constexpr enc::Sel16 s = enc::sel16_xor(0);
        return op(v, per_word(v, [](uint32_t w) { return permlanex16<s.lo, s.hi>(w, w); }));
#endif
    } else {
        return v;
    }
}

#endif  // __AMDGCN__

}  // namespace batchlas::sgp::amdgcn
