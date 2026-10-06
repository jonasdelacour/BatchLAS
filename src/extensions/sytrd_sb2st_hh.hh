#pragma once

// Stage-2 band -> tridiagonal by Householder bulge chasing, reflectors RETAINED so
// Z := Q2 Z is possible (the Givens sytrd_sb2st discards Q2). Plain sequential
// schedule, validated in playground/sb2st_hh_sequential.py. Reflector k acts on rows
// [start_k, start_k + len_k); Q = H_1 H_2 ... H_m (generation order), Q^H A Q = T,
// so the back-transform applies them in REVERSE generation order.
// evidence: docs/perf/sytrd.md#sytrd-the-householder-chase-against-the-givens-chase

#include "../util/internal-api.hh"
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstdint>
#include <vector>

namespace batchlas {
namespace internal {

// One stored reflector. `sweep` is kept because one sweep's reflectors act on disjoint
// row ranges (starts stride by kd, length <= kd) and can be applied concurrently.
struct Sb2stHhRefl {
    int32_t start;
    int32_t len;
    int32_t sweep;
};

// Replays the chase schedule on the host. It depends only on (n, kd), never on values,
// so it is identical for every batch item.
inline std::vector<Sb2stHhRefl> build_sb2st_hh_schedule(int32_t n, int32_t kd) {
    std::vector<Sb2stHhRefl> out;
    if (n <= 2 || kd <= 1) return out;

    for (int32_t st = 0; st + 2 < n; ++st) {
        int32_t r0 = st + 1;
        int32_t r1 = (st + kd < n - 1) ? (st + kd) : (n - 1);
        if (r1 <= r0) continue;

        // TYPE 1: annihilate column st below the subdiagonal.
        out.push_back(Sb2stHhRefl{r0, r1 - r0 + 1, st});

        // Chase the resulting bulge to the bottom of the band.
        while (true) {
            const int32_t p0 = r1 + 1;
            const int32_t p1 = (r1 + kd < n - 1) ? (r1 + kd) : (n - 1);
            if (p0 > p1) break;
            out.push_back(Sb2stHhRefl{p0, p1 - p0 + 1, st});
            r0 = p0;
            r1 = p1;
        }
    }
    return out;
}

inline int32_t sb2st_hh_num_reflectors(int32_t n, int32_t kd) {
    return static_cast<int32_t>(build_sb2st_hh_schedule(n, kd).size());
}

// Working half-bandwidth needed to hold transient bulge fill. A length-kd
// reflector applied symmetrically pushes fill up to kd rows below the band.
inline int32_t sb2st_hh_work_bandwidth(int32_t n, int32_t kd) {
    const int32_t want = 2 * kd;
    const int32_t cap = (n > 0) ? (n - 1) : 0;
    return (want < cap) ? want : cap;
}

// Band -> tridiagonal, retaining the reflectors.
//
//   ab_in      (kd+1) x n   lower band, read-only
//   ab_tri_out 2 x n        row 0 = diagonal, row 1 = *signed* subdiagonal,
//                           so build_phase_from_kd1_band consumes it unchanged
//   d_out/e_out             real diagonal and |subdiagonal|
//   v_out      kd x nrefl   reflector k in column k, v[0] = 1, zero-padded;
//                           nrefl == build_sb2st_hh_schedule(n, kd).size()
//   tau_out    nrefl
template <Backend B, typename T>
BATCHLAS_INTERNAL_API Event sytrd_sb2st_hh(Queue& ctx,
                                           const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                           const MatrixView<T, MatrixFormat::Dense>& ab_tri_out,
                                           const VectorView<typename base_type<T>::type>& d_out,
                                           const VectorView<typename base_type<T>::type>& e_out,
                                           const MatrixView<T, MatrixFormat::Dense>& v_out,
                                           const VectorView<T>& tau_out,
                                           Uplo uplo,
                                           int32_t kd,
                                           const Span<std::byte>& ws);

template <Backend B, typename T>
BATCHLAS_INTERNAL_API size_t sytrd_sb2st_hh_buffer_size(Queue& ctx, int32_t n, int32_t kd, int32_t batch);

// Splits the reflector list into maximal runs of pairwise-disjoint (hence commuting)
// reflectors; run w is [off[w], off[w+1]). The runs equal the chase sweeps, but are
// DERIVED from the schedule, so an unsound grouping cannot slip through.
inline std::vector<int32_t> build_sb2st_hh_wave_offsets(
    const std::vector<Sb2stHhRefl>& sched, int32_t n) {
    std::vector<int32_t> off;
    const int32_t nrefl = static_cast<int32_t>(sched.size());
    if (nrefl <= 0 || n <= 0) return off;

    // stamp[r] == run means row r is already claimed by the run being built.
    std::vector<int32_t> stamp(static_cast<size_t>(n), -1);
    int32_t run = 0;
    off.push_back(0);
    for (int32_t k = 0; k < nrefl; ++k) {
        const int32_t s = sched[k].start;
        const int32_t e = s + sched[k].len;
        bool overlaps = false;
        for (int32_t r = s; r < e && r < n; ++r) {
            if (stamp[r] == run) { overlaps = true; break; }
        }
        if (overlaps) { off.push_back(k); ++run; }
        for (int32_t r = s; r < e && r < n; ++r) stamp[r] = run;
    }
    off.push_back(nrefl);
    return off;
}

// Z := Q2 Z, reflectors in reverse generation order. `starts`/`lens` come from
// build_sb2st_hh_schedule, `waves` from build_sb2st_hh_wave_offsets (host side).
// All four spans must stay alive until the returned Event completes (not copied).
template <Backend B, typename T>
BATCHLAS_INTERNAL_API Event unmqr_hb2st(Queue& ctx,
                                        const MatrixView<T, MatrixFormat::Dense>& v_in,
                                        const VectorView<T>& tau_in,
                                        const MatrixView<T, MatrixFormat::Dense>& z_io,
                                        int32_t n,
                                        int32_t kd,
                                        Span<const int32_t> starts,
                                        Span<const int32_t> lens,
                                        Span<const int32_t> waves);

} // namespace internal
} // namespace batchlas
