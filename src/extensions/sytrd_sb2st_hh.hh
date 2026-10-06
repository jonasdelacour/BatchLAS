#pragma once

// Stage-2 band -> tridiagonal by Householder bulge chasing, reflectors RETAINED for Z := Q2 Z
// (the Givens sytrd_sb2st discards Q2). Reflector k acts on rows [start_k, start_k + len_k);
// Q = H_1 ... H_m in generation order, so the back-transform applies them in REVERSE.
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

// The chase schedule depends only on (n, kd), so one host replay serves every batch item.
inline std::vector<Sb2stHhRefl> build_sb2st_hh_schedule(int32_t n, int32_t kd) {
    std::vector<Sb2stHhRefl> out;
    if (n <= 2 || kd <= 1) return out;

    for (int32_t st = 0; st + 2 < n; ++st) {
        int32_t r0 = st + 1;
        int32_t r1 = (st + kd < n - 1) ? (st + kd) : (n - 1);
        if (r1 <= r0) continue;

        // Annihilate column st below the subdiagonal, then chase the bulge down the band.
        out.push_back(Sb2stHhRefl{r0, r1 - r0 + 1, st});
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

// A length-kd reflector applied symmetrically fills up to kd rows below the band: hold 2*kd.
inline int32_t sb2st_hh_work_bandwidth(int32_t n, int32_t kd) {
    const int32_t want = 2 * kd;
    const int32_t cap = (n > 0) ? (n - 1) : 0;
    return (want < cap) ? want : cap;
}

// ab_in: (kd+1) x n lower band, read-only. ab_tri_out: 2 x n, diagonal and SIGNED subdiagonal
// (build_phase_from_kd1_band consumes it unchanged). d_out/e_out: real diagonal, |subdiagonal|.
// v_out: kd x nrefl, reflector k in column k with v[0] = 1, zero-padded; tau_out: nrefl, where
// nrefl == build_sb2st_hh_schedule(n, kd).size().
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

// Maximal runs of pairwise-disjoint (commuting) reflectors, run w = [off[w], off[w+1]). They
// equal the sweeps but are DERIVED from the schedule, so an unsound grouping cannot slip in.
inline std::vector<int32_t> build_sb2st_hh_wave_offsets(
    const std::vector<Sb2stHhRefl>& sched, int32_t n) {
    std::vector<int32_t> off;
    const int32_t nrefl = static_cast<int32_t>(sched.size());
    if (nrefl <= 0 || n <= 0) return off;

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

// Z := Q2 Z. starts/lens/waves come from the two builders above; every span must outlive the
// returned Event (nothing is copied).
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
